import { useCallback, useEffect, useRef, useState } from 'react'
import { Map, NavigationControl, LngLatBounds, Popup } from 'maplibre-gl'

const STYLE_URL = 'https://tiles.openfreemap.org/styles/liberty'
const SATELLITE_TILES =
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'

// Growth is the hazard-relevant direction for glacial lakes, so it gets the
// warm end of the diverging scale.
const GROW = '#b4442e'
const FLAT = '#9aa3a8'
const SHRINK = '#3d5c7a'
const SELECT = '#111827'

const TOP_N = 15 // lakes given a numbered rank badge

// Column footprint for the 3D view. The overview sits near zoom 4, where one
// pixel spans ~3.7 km - a footprint sized to the actual lake would be smaller
// than a pixel and simply not draw. These are symbols, not lake outlines.
const COLUMN_RADIUS_M = 11000

/** Approximate a circle of `radiusM` around lon/lat as a polygon ring.
 *  MapLibre extrudes polygons only, so the risk columns need real geometry. */
function circlePolygon(lon, lat, radiusM, steps = 18) {
  const dLat = radiusM / 110540
  const dLon = radiusM / (111320 * Math.cos((lat * Math.PI) / 180))
  const ring = []
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * 2 * Math.PI
    ring.push([lon + dLon * Math.cos(t), lat + dLat * Math.sin(t)])
  }
  return [ring]
}

/** Point features -> squat polygons whose height encodes triage score. */
function riskColumns(data) {
  const feats = []
  for (const f of data.features || []) {
    const score = f.properties?.exposure_score
    if (!score || score <= 0) continue
    const [lon, lat] = f.geometry.coordinates
    feats.push({
      type: 'Feature',
      geometry: { type: 'Polygon', coordinates: circlePolygon(lon, lat, COLUMN_RADIUS_M) },
      properties: {
        id: f.properties.id,
        exposure_score: score,
        downstream_pop: f.properties.downstream_pop ?? 0,
        priority_rank: f.properties.priority_rank ?? null,
        // metres: the map spans ~2000 km, so a column has to be tens of km tall
        // to read as height rather than as a dot
        height: Math.max(score, 0) * 9000,
      },
    })
  }
  return { type: 'FeatureCollection', features: feats }
}

function addLayers(map, data) {
  if (map.getSource('lakes')) {
    map.getSource('lakes').setData(data)
    return
  }
  map.addSource('lakes', { type: 'geojson', data })

  // Base layer: every lake. Radius by area, colour by trend rate.
  map.addLayer({
    id: 'lakes-circles',
    type: 'circle',
    source: 'lakes',
    paint: {
      'circle-radius': [
        'interpolate', ['linear'], ['zoom'],
        4, ['interpolate', ['linear'], ['sqrt', ['max', ['get', 'area_last_ha'], 1]], 3, 2.5, 25, 7],
        10, ['interpolate', ['linear'], ['sqrt', ['max', ['get', 'area_last_ha'], 1]], 3, 7, 25, 20],
      ],
      'circle-color': [
        'interpolate', ['linear'], ['coalesce', ['get', 'slope_pct_per_yr'], 0],
        // matched to the real distribution (p10..p90 = -1.6..+1.2 %/yr)
        -2, SHRINK, -0.4, FLAT, 0.4, FLAT, 2, GROW,
      ],
      // weak trends stay visible but recede - hiding them would overstate certainty
      'circle-opacity': ['case', ['<', ['coalesce', ['get', 'r2'], 0], 0.5], 0.28, 0.85],
      'circle-stroke-width': 1,
      'circle-stroke-color': '#ffffff',
      'circle-stroke-opacity': 0.75,
    },
  })

  // Priority ring: makes the watchlist findable at a glance instead of
  // requiring the user to hunt for them among 839 dots.
  map.addLayer({
    id: 'lakes-priority-ring',
    type: 'circle',
    source: 'lakes',
    filter: ['all', ['!=', ['get', 'priority_rank'], null], ['<=', ['get', 'priority_rank'], TOP_N]],
    paint: {
      'circle-radius': [
        'interpolate', ['linear'], ['zoom'],
        4, 9, 10, 26,
      ],
      'circle-color': 'transparent',
      'circle-stroke-width': 2,
      'circle-stroke-color': GROW,
      'circle-stroke-opacity': 0.9,
    },
  })

  map.addLayer({
    id: 'lakes-rank-label',
    type: 'symbol',
    source: 'lakes',
    filter: ['all', ['!=', ['get', 'priority_rank'], null], ['<=', ['get', 'priority_rank'], TOP_N]],
    layout: {
      'text-field': ['to-string', ['get', 'priority_rank']],
      'text-size': 11,
      'text-offset': [0, -1.6],
      'text-allow-overlap': true,
    },
    paint: {
      'text-color': GROW,
      'text-halo-color': '#ffffff',
      'text-halo-width': 1.6,
    },
  })

  // Selection: deliberately loud. The previous version was a thin ring that
  // vanished among the other dots.
  map.addLayer({
    id: 'lakes-selected-halo',
    type: 'circle',
    source: 'lakes',
    filter: ['==', ['get', 'id'], ''],
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 4, 16, 10, 40],
      'circle-color': SELECT,
      'circle-opacity': 0.12,
      'circle-stroke-width': 3,
      'circle-stroke-color': SELECT,
    },
  })
}

/** Extruded risk columns + downstream flood corridors. Added once, then fed. */
function addRiskLayers(map, columns, flows) {
  if (!map.getSource('risk-cols')) {
    map.addSource('risk-cols', { type: 'geojson', data: columns })
    map.addLayer({
      id: 'risk-extrusion',
      type: 'fill-extrusion',
      source: 'risk-cols',
      layout: { visibility: 'none' },
      paint: {
        'fill-extrusion-color': [
          'interpolate', ['linear'], ['coalesce', ['get', 'downstream_pop'], 0],
          0, '#8fa3ad', 500, '#e0a03a', 5000, '#d2622c', 40000, '#8f1d1d',
        ],
        'fill-extrusion-height': ['get', 'height'],
        'fill-extrusion-base': 0,
        'fill-extrusion-opacity': 0.85,
      },
    })
  } else {
    map.getSource('risk-cols').setData(columns)
  }

  if (flows) {
    if (!map.getSource('flows')) {
      map.addSource('flows', { type: 'geojson', data: flows })
      map.addLayer({
        id: 'flow-lines',
        type: 'line',
        source: 'flows',
        layout: { visibility: 'none', 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': [
            'interpolate', ['linear'], ['coalesce', ['get', 'downstream_pop'], 0],
            0, '#9ec5d8', 500, '#e0a03a', 5000, '#d2622c', 40000, '#8f1d1d',
          ],
          'line-width': ['interpolate', ['linear'], ['zoom'], 4, 1.2, 10, 3.5],
          'line-opacity': 0.75,
        },
      }, 'risk-extrusion')
    } else {
      map.getSource('flows').setData(flows)
    }
  }
}

function applyData(map, data, fittedRef, fitKey, flows) {
  addLayers(map, data)
  try {
    addRiskLayers(map, riskColumns(data), flows)
  } catch { /* risk layers are optional - exposure data may not exist yet */ }
  // Refit only once per filter scope. Keyed on fitKey rather than a boolean so
  // a scope change that arrives before its data does not fit to the old set.
  if (fittedRef.current === fitKey) return
  let b = null
  for (const f of data.features) {
    const c = f.geometry.coordinates
    b = b ? b.extend(c) : new LngLatBounds(c, c)
  }
  if (b) map.fitBounds(b, { padding: 50, duration: 600 })
  fittedRef.current = fitKey
}


export default function LakeMap({ data, flows, selectedId, onSelect, fitKey }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const popupRef = useRef(null)
  const readyRef = useRef(false)
  const fittedRef = useRef(null)
  const pendingRef = useRef(null)
  const fitKeyRef = useRef(fitKey)

  const [satellite, setSatellite] = useState(false)
  const [iso, setIso] = useState(false)
  const flowsRef = useRef(flows)
  flowsRef.current = flows
  const select = useCallback((id) => onSelect?.(id), [onSelect])
  fitKeyRef.current = fitKey

  useEffect(() => {
    if (mapRef.current || !containerRef.current) return
    const map = new Map({
      container: containerRef.current,
      style: STYLE_URL,
      center: [86, 30],
      zoom: 4.6,
      // Every monitored lake sits between 73-95E and 27-36N, and the longest
      // downstream trace reaches ~26N. Panning to the Gulf or the Pacific shows
      // nothing and makes the watchlist look sparser than it is, so the camera
      // is fenced to the Himalayan arc plus the plains the floods drain into.
      minZoom: 4.2,
      maxBounds: [[69, 21], [99, 40]],
    })
    mapRef.current = map
    // dev-only handle so the rendered layer can be inspected/asserted against
    if (import.meta.env.DEV) window.__lakeMap = map
    popupRef.current = new Popup({ closeButton: false, closeOnClick: false, offset: 12 })
    map.addControl(new NavigationControl(), 'top-right')

    map.once('load', () => {
      readyRef.current = true
      // Data may have arrived before the style finished loading. Registering
      // once('load') from the data effect is unsafe - if load already fired it
      // never runs again - so the latest data is stashed and flushed here.
      if (pendingRef.current) {
        try {
          applyData(map, pendingRef.current, fittedRef, fitKeyRef.current, flowsRef.current)
          pendingRef.current = null
        } catch { /* retried on styledata */ }
      }
      const hit = ['lakes-circles', 'lakes-priority-ring']
      hit.forEach((layer) => {
        map.on('click', layer, (e) => select(e.features[0].properties.id))
        map.on('mousemove', layer, (e) => {
          map.getCanvas().style.cursor = 'pointer'
          const p = e.features[0].properties
          const rank = p.priority_rank ? `#${p.priority_rank} · ` : ''
          const slope = Number(p.slope_pct_per_yr)
          popupRef.current
            .setLngLat(e.features[0].geometry.coordinates)
            .setHTML(
              `<div style="font:11px system-ui;line-height:1.45">
                 <strong>${rank}${p.id}</strong><br/>
                 ${Number(p.area_last_ha).toFixed(1)} ha ·
                 <span style="color:${slope >= 0 ? GROW : SHRINK}">
                   ${slope >= 0 ? '+' : ''}${slope.toFixed(2)}%/yr</span><br/>
                 <span style="opacity:.65">R² ${Number(p.r2).toFixed(2)}</span>
               </div>`,
            )
            .addTo(map)
        })
        map.on('mouseleave', layer, () => {
          map.getCanvas().style.cursor = ''
          popupRef.current.remove()
        })
      })
    })

    return () => {
      map.remove()
      mapRef.current = null
      readyRef.current = false
    }
  }, [select])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !data?.features?.length) return
    // Do NOT gate on isStyleLoaded(): with some styles (OpenFreeMap here) it can
    // stay false indefinitely even though the style is usable, which silently
    // left the map with no layers at all. Try to attach, and retry on
    // 'styledata' if the style genuinely is not ready yet.
    const attach = () => {
      try {
        applyData(map, data, fittedRef, fitKey, flowsRef.current)
        return true
      } catch {
        return false
      }
    }
    if (!attach()) {
      pendingRef.current = data
      const retry = () => {
        if (pendingRef.current && attach()) {
          pendingRef.current = null
          map.off('styledata', retry)
        }
      }
      map.on('styledata', retry)
      return () => map.off('styledata', retry)
    }
    // flows is a dependency because it arrives on its own request: when the
    // lakes land first, the corridors would otherwise never be added. Re-running
    // is safe - sources are updated in place and the fit is keyed on fitKey.
  }, [data, fitKey, flows])

  // Selection: highlight AND fly to it, so clicking a table row actually takes
  // you somewhere rather than silently recolouring an off-screen dot.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !readyRef.current || !map.getLayer('lakes-selected-halo')) return
    map.setFilter('lakes-selected-halo', ['==', ['get', 'id'], selectedId ?? ''])
    if (!selectedId || !data?.features) return
    const f = data.features.find((x) => x.properties.id === selectedId)
    if (f) {
      map.flyTo({
        center: f.geometry.coordinates,
        zoom: Math.max(map.getZoom(), 10),
        duration: 900,
      })
    }
  }, [selectedId, data])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const apply = () => {
      if (satellite && !map.getSource('sat')) {
        map.addSource('sat', {
          type: 'raster', tiles: [SATELLITE_TILES], tileSize: 256,
          attribution: 'Imagery &copy; Esri',
        })
        map.addLayer({ id: 'sat-layer', type: 'raster', source: 'sat' },
          map.getStyle().layers[0]?.id)
      }
      if (map.getLayer('sat-layer')) {
        map.setLayoutProperty('sat-layer', 'visibility', satellite ? 'visible' : 'none')
      }
    }
    if (readyRef.current) apply()
    else map.once('load', apply)
  }, [satellite])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const flat = ['lakes-circles', 'lakes-priority-ring', 'lakes-rank-label']
    const three = ['risk-extrusion', 'flow-lines']

    // Do NOT gate on readyRef: it is set from map.once('load'), which with this
    // style can never fire even though the style is perfectly usable - the same
    // trap that once left the map with zero lake layers. Attempt the change and
    // retry on styledata until the layers actually exist.
    const camera = { pitch: iso ? 55 : 0, bearing: iso ? -22 : 0 }
    let settle = null
    let retry = null

    const apply = () => {
      const present = [...flat, ...three].filter((id) => map.getLayer(id))
      if (!present.length) return false
      present.forEach((id) =>
        map.setLayoutProperty(
          id,
          'visibility',
          three.includes(id) === iso ? 'visible' : 'none',
        ),
      )
      map.easeTo({ ...camera, duration: 700 })
      // applyData's fitBounds animates and can land after this, resetting pitch
      // to 0. Re-assert once, when the camera next settles.
      settle = () => {
        if (Math.abs(map.getPitch() - camera.pitch) > 1) map.jumpTo(camera)
        settle = null
      }
      map.once('moveend', settle)
      return true
    }

    if (!apply()) {
      retry = () => { if (apply()) { map.off('styledata', retry); retry = null } }
      map.on('styledata', retry)
    }
    // Both handlers MUST be torn down. A settle left over from the previous
    // mode still carries that mode's pitch, and fires on this mode's easeTo -
    // which is exactly how 3D kept snapping back to flat.
    return () => {
      if (settle) map.off('moveend', settle)
      if (retry) map.off('styledata', retry)
    }
  }, [iso, data, flows])

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />

      <button
        onClick={() => setIso((v) => !v)}
        className="absolute top-3 left-[5.2rem] border border-line bg-paper px-2 py-1 font-sans text-[11px] text-ink shadow-sm hover:bg-canvas"
      >
        {iso ? '2D' : '3D exposure'}
      </button>

      <button
        onClick={() => setSatellite((s) => !s)}
        className="absolute top-3 left-3 border border-line bg-paper px-2 py-1 font-sans text-[11px] text-ink shadow-sm hover:bg-canvas"
      >
        {satellite ? 'Map' : 'Satellite'}
      </button>

      {/* Two lines. The map is the thing being read; the legend should not
          compete with it. Detail lives in tooltips and the Method panel. */}
      <div className="absolute bottom-3 left-3 border border-line bg-paper px-2.5 py-1.5 font-sans text-[10px] text-muted shadow-sm">
        {iso ? (
          <div>
            taller = more urgent · redder = more people downstream · lines = flood path
          </div>
        ) : (
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: SHRINK }} />
            shrinking
            <span className="ml-1.5 inline-block h-2 w-2 rounded-full" style={{ background: FLAT }} />
            stable
            <span className="ml-1.5 inline-block h-2 w-2 rounded-full" style={{ background: GROW }} />
            growing
            <span className="ml-2 opacity-70">· numbered = top {TOP_N} · size = area</span>
          </div>
        )}
      </div>
    </div>
  )
}
