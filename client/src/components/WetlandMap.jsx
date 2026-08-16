import { useEffect, useRef } from 'react'
import { Map, NavigationControl } from 'maplibre-gl'

// OpenFreeMap's "liberty" style is free, needs no API key, and already ships
// a fill-extrusion "building-3d" layer (kicks in from zoom 14) — that's what
// gives the 2.5D look. Swap for a MapTiler style + VITE_MAPTILER_KEY later
// if we need higher-detail imagery/labels.
const STYLE_URL = 'https://tiles.openfreemap.org/styles/liberty'

// Central Kolkata, near the Hooghly — dense buildings so the 3D effect is
// visible out of the box. Swap for the demo wetland's centroid once real
// boundary data is wired up.
const DEMO_CENTER = [88.3639, 22.5726]

export default function WetlandMap() {
  const containerRef = useRef(null)
  const mapRef = useRef(null)

  useEffect(() => {
    if (mapRef.current || !containerRef.current) return

    mapRef.current = new Map({
      container: containerRef.current,
      style: STYLE_URL,
      center: DEMO_CENTER,
      zoom: 16,
      pitch: 55,
      bearing: -17,
    })

    mapRef.current.addControl(
      new NavigationControl({ visualizePitch: true }),
      'top-right',
    )

    return () => {
      mapRef.current?.remove()
      mapRef.current = null
    }
  }, [])

  return <div ref={containerRef} className="h-full w-full" />
}
