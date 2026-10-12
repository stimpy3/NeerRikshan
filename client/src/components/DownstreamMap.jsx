import { useEffect, useRef } from 'react'
import { Map, LngLatBounds, Marker, Popup, NavigationControl } from 'maplibre-gl'

const STYLE_URL = 'https://tiles.openfreemap.org/styles/liberty'
const LAKE = '#2e6fb4'
const FLOW = '#8f1d1d'

/**
 * The traced flood path for one lake, with the populated clusters along it.
 *
 * The head-count in the panel above answers "how many"; this answers "where",
 * which is the part someone deciding whether to drive up a valley actually
 * needs. Settlement *names* are left to the basemap, which already labels them -
 * our markers only assert that people are there, because GHSL is a population
 * grid and cannot name a town.
 *
 * Marker size scales with population so the largest cluster is findable without
 * reading every label.
 */
export default function DownstreamMap({ path, settlements, lake }) {
  const boxRef = useRef(null)
  const mapRef = useRef(null)

  useEffect(() => {
    if (!boxRef.current || !path?.length) return undefined
    const map = new Map({
      container: boxRef.current,
      style: STYLE_URL,
      center: path[0],
      zoom: 7,
      attributionControl: false,
    })
    mapRef.current = map
    map.addControl(new NavigationControl({ showCompass: false }), 'top-right')

    // Same attempt-and-retry as the main map: map.once('load') can never fire
    // with this style, and gating on it leaves an empty map.
    const draw = () => {
      if (map.getSource('flow')) return true
      try {
        map.addSource('flow', {
          type: 'geojson',
          data: { type: 'Feature', geometry: { type: 'LineString', coordinates: path } },
        })
        map.addLayer({
          id: 'flow-casing',
          type: 'line',
          source: 'flow',
          layout: { 'line-cap': 'round', 'line-join': 'round' },
          paint: { 'line-color': '#ffffff', 'line-width': 6, 'line-opacity': 0.9 },
        })
        map.addLayer({
          id: 'flow-line',
          type: 'line',
          source: 'flow',
          layout: { 'line-cap': 'round', 'line-join': 'round' },
          paint: { 'line-color': FLOW, 'line-width': 2.5 },
        })
      } catch {
        return false
      }

      new Marker({ color: LAKE })
        .setLngLat(path[0])
        .setPopup(new Popup({ offset: 12 }).setText(`${lake} — source`))
        .addTo(map)

      const biggest = Math.max(1, ...(settlements || []).map((s) => s.pop))
      for (const s of settlements || []) {
        const el = document.createElement('div')
        const size = 8 + 14 * Math.sqrt(s.pop / biggest)
        el.style.cssText = `width:${size}px;height:${size}px;border-radius:50%;
          background:${FLOW};opacity:.75;border:1.5px solid #fff;cursor:pointer`
        const label = s.name
          ? `${s.name} — ~${s.pop.toLocaleString()} people · ${s.km} km downstream`
          : `~${s.pop.toLocaleString()} people · ${s.km} km downstream`
        new Marker({ element: el })
          .setLngLat([s.lon, s.lat])
          .setPopup(new Popup({ offset: 8 }).setText(label))
          .addTo(map)
      }

      let b = new LngLatBounds(path[0], path[0])
      path.forEach((c) => b.extend(c))
      map.fitBounds(b, { padding: 28, duration: 0 })
      return true
    }

    if (!draw()) {
      const retry = () => { if (draw()) map.off('styledata', retry) }
      map.on('styledata', retry)
    }
    return () => { map.remove(); mapRef.current = null }
  }, [path, settlements, lake])

  if (!path?.length) {
    return (
      <div className="border border-dashed border-line px-3 py-4 font-sans text-[11px] text-muted">
        No downstream trace for this lake — paths are computed only for lakes with
        confidently growing area.
      </div>
    )
  }

  // Named places, in downstream order, one entry each. The same town often
  // spans several population cells, so dedupe by name and keep the nearest hit.
  const places = []
  for (const s of [...(settlements || [])].sort((a, b) => a.km - b.km)) {
    if (!s.name) continue
    const seen = places.find((p) => p.name === s.name)
    if (seen) seen.pop = Math.max(seen.pop, s.pop)
    else places.push({ name: s.name, km: s.km, pop: s.pop })
  }
  const unnamed = (settlements || []).filter((s) => !s.name).length

  return (
    <div>
      <div ref={boxRef} className="h-[260px] w-full border border-line" />

      {places.length > 0 && (
        <div className="mt-1.5 border border-line bg-canvas px-2 py-1.5">
          <div className="font-sans text-[10px] uppercase tracking-wide text-muted">
            Settlements in the path
          </div>
          <div className="mt-0.5 font-sans text-[11px] leading-relaxed text-ink">
            {places.map((p, i) => (
              <span key={p.name}>
                {i > 0 && <span className="text-muted"> → </span>}
                <span className="font-semibold">{p.name}</span>
                <span className="text-muted">
                  {' '}
                  {p.km}&nbsp;km
                </span>
              </span>
            ))}
          </div>
          {unnamed > 0 && (
            <div className="mt-0.5 font-sans text-[10px] text-muted">
              + {unnamed} inhabited cluster{unnamed > 1 ? 's' : ''} with no named
              place within 12&nbsp;km
            </div>
          )}
        </div>
      )}

      <p className="mt-1 font-sans text-[10px] leading-snug text-muted">
        Red line = path water would follow downhill from the lake (HydroSHEDS
        terrain routing). Circles = inhabited clusters on that path, sized by
        population; click for numbers. Names are the nearest GeoNames settlement
        to each cluster, not a claim that the whole town is in the flood corridor.
      </p>
    </div>
  )
}
