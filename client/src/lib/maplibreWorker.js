// MapLibre 6 finds its web worker at run time as `maplibre-gl-worker.mjs`
// next to its own module. A production build folds maplibre into the app
// bundle and never emits that file, so the worker 404s and every vector
// layer - basemap labels and the lake dots alike - silently fails to draw
// (raster hillshade still shows, which makes it look half-working). Dev mode
// serves maplibre unbundled, which is why this only breaks once deployed.
//
// `?worker&url` has Vite bundle the worker together with the shared chunk it
// imports, emit it as its own file, and hand back that file's URL. Production
// only: in dev maplibre's own lookup already resolves, and the dev server's
// worker wrapper URL does not load as a maplibre worker.
import { setWorkerUrl } from 'maplibre-gl'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'

if (import.meta.env.PROD) setWorkerUrl(workerUrl)
