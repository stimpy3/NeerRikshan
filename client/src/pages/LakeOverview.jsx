import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import LakeMap from '../components/LakeMap'
import { getLakeStats, getLakes, getFlowPaths } from '../lib/api'

const GROW = '#b4442e'
const SHRINK = '#3d5c7a'
const SUBREGIONS = ['All', 'Central Himalaya', 'Eastern Himalaya', 'Western Himalaya']

// Colour carries the decision, not the magnitude: a reader should be able to see
// which lakes to open first without reading a single number.
const BANDS = {
  inspect_first: { label: 'Inspect first', dot: '#8f1d1d',
    help: 'Confidently growing, and 10,000+ people live along the path water would take downstream' },
  monitor: { label: 'Monitor', dot: '#d2622c',
    help: 'Confidently growing, with 1,000+ people downstream' },
  watch: { label: 'Watch', dot: '#e0a03a',
    help: 'Growing more slowly, with 1,000+ people downstream' },
  low_exposure: { label: 'Few below', dot: '#8fa3ad',
    help: 'Growing, but fewer than 1,000 people downstream' },
  unassessed: { label: 'Not traced', dot: '#c9ced1',
    help: 'No detectable growth, so no downstream trace was run' },
}

function peopleShort(n) {
  if (n == null) return '—'
  if (n >= 1000) return `${Math.round(n / 1000)}k`
  return String(n)
}

// Four ways to read the same list. Seven was a menu, not a choice.
const SORTS = [
  { key: 'screening_priority', label: 'Priority' },
  { key: 'downstream_pop', label: 'People downstream' },
  { key: 'priority', label: 'Growth rate' },
  { key: 'area_last_ha', label: 'Lake size' },
]

function Stat({ value, label, hint, accent }) {
  return (
    <div className="border-r border-line px-4 py-2 last:border-r-0">
      <div className="font-condensed text-2xl leading-none" style={{ color: accent || '#1d1f20' }}>
        {value ?? '—'}
      </div>
      <div className="mt-0.5 font-sans text-[11px] text-ink">{label}</div>
      {hint && <div className="font-sans text-[10px] text-muted">{hint}</div>}
    </div>
  )
}

export default function LakeOverview() {
  const navigate = useNavigate()
  const [stats, setStats] = useState(null)
  const [rows, setRows] = useState([])
  const [hoverId, setHoverId] = useState(null)
  const [subregion, setSubregion] = useState('All')
  const [attentionOnly, setAttentionOnly] = useState(true)
  const [sort, setSort] = useState('screening_priority')
  const [showMethod, setShowMethod] = useState(false)
  const [error, setError] = useState(null)
  // Flood corridors for the 3D view. Optional: absent until
  // scripts/trace_downstream.py has run, and the map degrades to 2D without it.
  const [flows, setFlows] = useState(null)

  useEffect(() => {
    getFlowPaths({ limit: 120 })
      .then(setFlows)
      .catch(() => setFlows(null))
  }, [])

  useEffect(() => {
    const params = { min_r2: 0, sort, limit: 2000 }
    if (subregion !== 'All') params.subregion = subregion
    Promise.all([getLakeStats(), getLakes(params)])
      .then(([s, r]) => { setStats(s); setRows(r); setError(null) })
      .catch((e) => {
        // FastAPI 422 returns `detail` as an array of objects; rendering that
        // straight into JSX crashes React and blanks the whole page.
        const d = e?.response?.data?.detail
        setError(typeof d === 'string' ? d : d ? JSON.stringify(d) : e.message ?? 'failed to load')
      })
  }, [subregion, sort])

  // Clicking anywhere - row or map - opens the full page. A cramped inline
  // panel duplicating a real detail view helps nobody.
  const open = useCallback((id) => navigate(`/lakes/${id}`), [navigate])

  // Map data is derived from the same filtered rows the table renders, so the
  // two can never disagree. Previously the map fetched its own fixed set and
  // ignored the filters entirely.
  // Showing 191 rows when only 94 carry a band meant ~100 lines of "—".
  const visible = useMemo(
    () => (attentionOnly ? rows.filter((r) => (r.priority ?? 0) > 0) : rows),
    [rows, attentionOnly],
  )

  const geo = useMemo(() => ({
    type: 'FeatureCollection',
    features: visible
      .filter((r) => r.lon != null && r.lat != null)
      .map((r) => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [r.lon, r.lat] },
        properties: {
          id: r.id, subr: r.subr, r2: r.r2,
          pct_change: r.pct_change,
          slope_pct_per_yr: r.slope_pct_per_yr,
          area_last_ha: r.area_last_ha,
          priority_rank: r.priority_rank ?? null,
          exposure_score: r.exposure_score ?? null,
          downstream_pop: r.downstream_pop ?? null,
          downstream_mw: r.downstream_mw ?? null,
        },
      })),
  }), [visible])

  const watchlist = useMemo(() => rows.filter((r) => (r.priority ?? 0) > 0), [rows])

  if (error) {
    return (
      <div className="p-6 font-sans text-sm">
        <div className="mb-2 font-semibold text-ink">Lake data not available</div>
        <div className="text-muted">{error}</div>
      </div>
    )
  }

  return (
    <div className="flex h-screen flex-col bg-paper">
      <header className="flex flex-wrap items-baseline gap-3 border-b border-line px-4 py-2">
        <h1 className="font-condensed text-xl text-ink">Himalayan Glacial Lake Watch</h1>
        <p className="font-sans text-[11px] text-muted">
          Sentinel-2, 2019–2025 · lakes ≥10 ha
        </p>
        <button
          onClick={() => setShowMethod((v) => !v)}
          className="ml-auto border border-line px-2 py-0.5 font-sans text-[11px] text-muted hover:bg-canvas"
        >
          {showMethod ? 'Hide' : 'Method & limits'}
        </button>
      </header>

      {showMethod && (
        <div className="border-b border-line bg-canvas px-4 py-2 font-sans text-[11px] leading-relaxed text-ink">
          <strong>What this fills.</strong> Glacial lake inventories are episodic snapshots — the
          Hi-MAG baseline used here <em>ends in 2017</em>. This extends it to 2025 with a per-lake
          annual series, so growth can be ranked rather than rediscovered.
          <br />
          <strong>How.</strong> A U-Net segments melt-season Sentinel-2 composites, trained on 2022
          imagery with epoch-matched 10 m labels, then frozen and applied to every year. On 100
          held-out lakes: <strong>IoU 0.91 / recall 0.96</strong>, versus 0.32 for an MNDWI
          threshold — which cannot separate snow and ice from water, being algebraically the same
          index. Windows are sized to each lake, because a fixed 1.28 km frame clipped every lake
          over 100 ha and made their growth unmeasurable.
          <br />
          <strong>Screening score</strong> = the <em>lower 95% confidence bound</em> on hectares
          gained per year × log₁₀(people in the downstream flood corridor), traced 100 km along
          HydroSHEDS flow direction. A lake is ranked only if its trend is distinguishable from
          zero. Downstream infrastructure is shown but deliberately <em>not</em> folded in.
          <br />
          <strong className="text-[#b4442e]">This is not a GLOF forecast.</strong> It combines two
          things we observe — how fast a lake is growing, and who lives along the path water would
          take if it drained. It says nothing about <em>whether</em> a lake will burst. That would
          require moraine-dam geometry and freeboard, ice-core degradation, lake volume and depth
          (we measure surface area only), avalanche and slope-failure potential above the lake,
          trigger events, and breach modelling — none of which are measured here. A high score
          means <em>worth looking at</em>; a low score is not a safety certificate.
          <br />
          <strong className="text-[#b4442e]">Limits.</strong> Only{' '}
          {stats?.lakes_detectable ?? '—'} of {stats?.lakes_clean ?? '—'} lakes have a trend whose
          confidence interval excludes zero; the other {stats?.lakes_no_trend ?? '—'} have a
          six-year signal smaller than their own scatter and are <em>not</em> ranked. Absolute
          areas read ~15% high (proportionally, so rates are unaffected). Six years is short:
          growth measured here runs ~25% below an independent inventory spanning 2016–2023, partly
          because growth slowed after 2022. Lakes ≥10 ha only — about 17% of the inventory.
        </div>
      )}

      <div className="flex flex-wrap border-b border-line">
        {/* Four numbers, read left to right as one sentence: we watched this
            many, this many are really growing, these need a look. Six competed
            with each other and none of them landed. */}
        <Stat value={stats?.lakes_analysed} label="Monitored" hint="Himalaya, ≥10 ha" />
        <Stat value={stats?.growing} label="Confidently growing" hint="of 152 with a real trend"
          accent={GROW} />
        <Stat value={stats?.inspect_first} label="Inspect first" hint="≥10k people downstream"
          accent="#8f1d1d" />
        <Stat value={stats?.monitor} label="Monitor" hint="≥1k people downstream"
          accent="#d2622c" />
      </div>

      <div className="flex min-h-0 flex-1">
        <div className="flex w-[400px] flex-col border-r border-line">
          <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-2 font-sans text-[11px]">
            <select value={subregion} onChange={(e) => setSubregion(e.target.value)}
              className="border border-line bg-paper px-1.5 py-1 text-ink">
              {SUBREGIONS.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <select value={sort} onChange={(e) => setSort(e.target.value)}
              className="border border-line bg-paper px-1.5 py-1 text-ink">
              {SORTS.map((s) => <option key={s.key} value={s.key}>Sort: {s.label}</option>)}
            </select>
            <label className="flex items-center gap-1.5 text-muted"
              title="Hide lakes whose six-year change cannot be distinguished from their own year-to-year noise">
              <input type="checkbox" checked={attentionOnly}
                onChange={(e) => setAttentionOnly(e.target.checked)} />
              needs attention
            </label>
            <span className="ml-auto text-muted">{visible.length} shown</span>
          </div>

          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-line bg-canvas px-3 py-1.5 font-sans text-[10px] text-muted">
            {['inspect_first', 'monitor', 'watch', 'low_exposure'].map((k) => (
              <span key={k} className="flex items-center gap-1" title={BANDS[k].help}>
                <span className="inline-block h-2 w-2 rounded-full"
                  style={{ background: BANDS[k].dot }} />
                {BANDS[k].label}
              </span>
            ))}
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto">
            <div className="sticky top-0 z-10 grid grid-cols-[26px_10px_1fr_46px_52px_46px] border-b border-line bg-paper px-3 py-1 font-sans text-[10px] uppercase tracking-wide text-muted">
              <span>#</span><span /><span>Lake</span><span className="text-right">ha</span>
              <span className="text-right"
                title="Hectares gained per year — the lower 95% confidence bound, i.e. the growth we can defend rather than the best fit">
                ha/yr
              </span>
              <span className="text-right"
                title="People living along the path water would take if this lake drained, traced 100 km downstream. ⚡ marks power generation in that path.">
                people
              </span>
            </div>
            {visible.map((r) => {
              const rising = (r.slope_pct_per_yr ?? 0) >= 0
              return (
                <button
                  key={r.id}
                  onClick={() => open(r.id)}
                  onMouseEnter={() => setHoverId(r.id)}
                  onMouseLeave={() => setHoverId(null)}
                  title="Open full view"
                  className={`grid w-full grid-cols-[26px_10px_1fr_46px_52px_46px] items-center border-b border-line px-3 py-1.5 text-left font-sans text-[11px] hover:bg-canvas ${
                    r.id === hoverId ? 'bg-canvas' : ''
                  }`}
                >
                  <span className="font-condensed text-[13px]"
                    style={{ color: r.priority_rank && r.priority_rank <= 15 ? GROW : '#9aa3a8' }}>
                    {r.priority_rank ?? '·'}
                  </span>
                  <span
                    title={BANDS[r.screening_band]?.label ?? 'No detectable growth'}
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ background: BANDS[r.screening_band]?.dot ?? 'transparent' }}
                  />
                  <span className="truncate">
                    <span className="text-ink">{r.id}</span>
                    <span className="ml-1 text-muted">{r.subr?.split(' ')[0]}</span>
                  </span>
                  <span className="text-right text-muted">{r.area_last_ha?.toFixed(1)}</span>
                  <span className="text-right font-semibold" style={{ color: rising ? GROW : SHRINK }}>
                    {r.priority > 0 ? `+${r.priority.toFixed(1)}` : '—'}
                  </span>
                  <span className="text-right" style={{ color: (r.downstream_pop ?? 0) >= 10000 ? '#8f1d1d' : '#6b7378' }}>
                    {peopleShort(r.downstream_pop)}
                    {r.downstream_mw > 0 && <span title={`${r.downstream_mw} MW downstream`}> ⚡</span>}
                  </span>
                </button>
              )
            })}
            {!visible.length && (
              <div className="px-3 py-4 font-sans text-[11px] text-muted">
                No lakes match these filters.
              </div>
            )}
          </div>

          <div className="border-t border-line px-3 py-1.5 font-sans text-[10px] text-muted">
            Click any lake — list or map — to open imagery, trend and caveats.
          </div>
        </div>

        <div className="min-w-0 flex-1">
          <LakeMap
          flows={flows} data={geo} selectedId={hoverId} onSelect={open} fitKey={subregion} />
        </div>
      </div>
    </div>
  )
}
