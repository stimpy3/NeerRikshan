import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import LakeFrames from '../components/LakeFrames'
import DownstreamMap from '../components/DownstreamMap'

import LakeTrend from '../components/LakeTrend'
import { getLake, getLakes } from '../lib/api'

const GROW = '#b4442e'
const SHRINK = '#3d5c7a'
// GHSL degree-of-urbanisation classes, collapsed to words a reader knows.
const SMOD = {
  10: 'Open country', 11: 'Very low density', 12: 'Low density', 13: 'Rural cluster',
  21: 'Suburban', 22: 'Semi-dense town', 23: 'Dense town', 30: 'City',
}
const BAND_DOT = {
  inspect_first: '#8f1d1d', monitor: '#d2622c', watch: '#e0a03a',
  low_exposure: '#8fa3ad', unassessed: '#c9ced1',
}

function Metric({ label, value, hint, accent }) {
  return (
    <div className="border border-line bg-paper px-3 py-2">
      <div className="font-sans text-[10px] uppercase tracking-wide text-muted">{label}</div>
      <div className="font-condensed text-2xl leading-tight" style={{ color: accent || '#1d1f20' }}>
        {value}
      </div>
      {hint && <div className="font-sans text-[10px] text-muted">{hint}</div>}
    </div>
  )
}

export default function LakeDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [lake, setLake] = useState(null)
  const [ranked, setRanked] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    setLake(null)
    getLake(id).then(setLake).catch((e) => setError(e.message))
  }, [id])

  useEffect(() => {
    getLakes({ sort: 'screening_priority', min_r2: 0.5, limit: 2000 })
      .then((rows) => setRanked(rows.filter((r) => (r.priority ?? 0) > 0)))
      .catch(() => setRanked([]))
  }, [])

  const { prev, next, pos } = useMemo(() => {
    const i = ranked.findIndex((r) => r.id === id)
    return {
      pos: i >= 0 ? i + 1 : null,
      prev: i > 0 ? ranked[i - 1] : null,
      next: i >= 0 && i < ranked.length - 1 ? ranked[i + 1] : null,
    }
  }, [ranked, id])

  if (error) {
    return (
      <div className="p-6 font-sans text-sm">
        <Link to="/" className="text-muted underline">← back</Link>
        <div className="mt-2 text-ink">{error}</div>
      </div>
    )
  }
  if (!lake) {
    return <div className="p-6 font-sans text-sm text-muted">Loading {id}…</div>
  }

  const rising = (lake.slope_pct_per_yr ?? 0) >= 0
  const accent = rising ? GROW : SHRINK
  const years = Object.entries(lake.raw_years || {}).sort()
  const excluded = years.filter(([, v]) => v.flag !== 'ok')
  // patch_frac near the saturation cutoff means the lake is outgrowing its
  // analysis window, so the area is closer to a lower bound than a measurement
  const nearSaturation = years.filter(([, v]) => (v.patch_frac ?? 0) > 0.5)

  return (
    <div className="min-h-screen bg-paper">
      <header className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-2">
        <Link to="/" className="font-sans text-[11px] text-muted underline">← All lakes</Link>
        {lake.priority_rank && (
          <span className="font-condensed text-lg" style={{ color: GROW }}>
            #{lake.priority_rank}
          </span>
        )}
        <h1 className="font-condensed text-xl text-ink">{lake.id}</h1>
        <span className="font-sans text-[11px] text-muted">
          {lake.subr} · {lake.lat?.toFixed(3)}°N {lake.lon?.toFixed(3)}°E
        </span>

        <div className="ml-auto flex items-center gap-1">
          <button
            disabled={!prev}
            onClick={() => prev && navigate(`/lakes/${prev.id}`)}
            className="border border-line px-2 py-1 font-sans text-[11px] text-ink disabled:opacity-30 hover:bg-canvas"
          >
            ← higher priority
          </button>
          {pos && (
            <span className="px-1 font-sans text-[11px] text-muted">
              {pos} / {ranked.length}
            </span>
          )}
          <button
            disabled={!next}
            onClick={() => next && navigate(`/lakes/${next.id}`)}
            className="border border-line px-2 py-1 font-sans text-[11px] text-ink disabled:opacity-30 hover:bg-canvas"
          >
            lower priority →
          </button>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1400px] gap-4 p-4 lg:grid-cols-[420px_1fr]">
        {/* LEFT: imagery, given real room this time */}
        <section>
          <h2 className="mb-1 font-sans text-[11px] uppercase tracking-wide text-muted">
            Sentinel-2 imagery by year
          </h2>
          <LakeFrames lakeId={lake.id} series={lake.series} />
          <p className="mt-2 font-sans text-[11px] leading-relaxed text-muted">
            Toggle <strong className="text-ink">outline</strong> to see what the model actually
            segmented. If the outline does not follow visible water, that year's area is not
            trustworthy no matter how clean the trend looks.
          </p>

          {/* Fills the dead space under the imagery, and pairs the "how many
              people" number on the right with "where they are". */}
          <h2 className="mt-4 mb-1 font-sans text-[11px] uppercase tracking-wide text-muted">
            Downstream path &amp; settlements
          </h2>
          <DownstreamMap
            path={lake.flow_path}
            settlements={lake.settlements}
            lake={lake.id}
          />
        </section>

        {/* RIGHT: numbers and caveats */}
        <section className="space-y-4">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Metric label="Area now" value={`${lake.area_last_ha} ha`}
              hint={`${lake.first_year}→${lake.last_year}`} />
            <Metric label="Change" accent={accent}
              value={`${lake.pct_change >= 0 ? '+' : ''}${lake.pct_change}%`} />
            <Metric label="Gained" accent={accent}
              value={`${lake.abs_change_ha >= 0 ? '+' : ''}${lake.abs_change_ha} ha`}
              hint={`${lake.slope_ha_per_yr} ha/yr`} />
            <Metric label="Confidence" value={lake.r2?.toFixed(2)}
              hint={lake.r2 >= 0.5 ? 'usable trend' : 'weak — treat as noise'}
              accent={lake.r2 >= 0.5 ? null : GROW} />
          </div>

          {/* Why this lake is on the list, in one sentence, before any table. */}
          {lake.screening_summary && (
            <div className="border border-line bg-canvas px-3 py-2">
              <div className="font-sans text-[10px] uppercase tracking-wide text-muted">
                Why this lake is listed
              </div>
              <div className="mt-0.5 font-sans text-[12px] text-ink">
                {lake.screening_summary}
              </div>
              {lake.screening_label && (
                <div className="mt-1 inline-block border border-line px-1.5 py-0.5 font-sans text-[10px]"
                  style={{ color: BAND_DOT[lake.screening_band] }}>
                  {lake.screening_label}
                </div>
              )}
            </div>
          )}

          {/* Downstream exposure. Absent for lakes that were never traced - only
              confidently growing lakes are, since tracing 850 would be wasted on
              lakes with no trend to act on. */}
          {lake.downstream_pop != null && (
            <div>
              <h2 className="mb-1 font-sans text-[11px] uppercase tracking-wide text-muted">
                If this lake drained — who is downstream
              </h2>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <Metric label="People" value={lake.downstream_pop.toLocaleString()}
                  hint={`within ${Math.round(lake.flow_path_km ?? 100)} km`}
                  accent={lake.downstream_pop >= 10000 ? '#8f1d1d' : null} />
                <Metric label="Power plants" value={lake.downstream_plants ?? 0}
                  hint={lake.downstream_mw ? `${lake.downstream_mw.toLocaleString()} MW` : 'none found'}
                  accent={lake.downstream_mw > 0 ? '#8f1d1d' : null} />
                <Metric label="Flood path" value={`${Math.round(lake.flow_path_km ?? 0)} km`}
                  hint="traced along terrain" />
                <Metric label="Settlement" value={SMOD[lake.settlement_class] ?? '—'}
                  hint="densest class in path" />
              </div>
              <p className="mt-2 border-l-4 px-3 py-2 font-sans text-[11px] leading-relaxed"
                style={{ borderColor: '#8f1d1d', background: '#00000006' }}>
                <strong style={{ color: '#8f1d1d' }}>This is exposure, not a forecast.</strong>{' '}
                The path is where water would go if this lake released, traced downhill through
                the terrain, and the count is who lives along it. It says nothing about{' '}
                <em>whether</em> this lake will burst — that needs the dam's structure and
                freeboard, the lake's depth and volume, ice and rockfall hazard above it, and a
                trigger. None of those are measured here. Read this as
                “worth a closer look”, never as “safe” or “about to fail”.
              </p>
            </div>
          )}

          <div
            className="border-l-4 px-3 py-2 font-sans text-[11px] leading-relaxed"
            style={{ borderColor: rising ? GROW : SHRINK, background: '#00000006' }}
          >
            {rising ? (
              <>
                <strong style={{ color: GROW }}>Growing — this is the hazard direction.</strong>{' '}
                Glacial lakes expand as glaciers retreat and meltwater pools behind moraine dams.
                A large enough lake can breach its dam and cause a glacial lake outburst flood
                downstream. Growth is why this lake is ranked, not the size itself.
              </>
            ) : (
              <>
                <strong style={{ color: SHRINK }}>Shrinking — not ranked as a hazard.</strong>{' '}
                Shrinkage is usually gradual drainage or measurement noise. A sudden large drop
                can indicate an outburst that already happened, which is worth checking against
                the imagery, but it is not a forward-looking risk.
              </>
            )}
          </div>

          <div>
            <h2 className="mb-1 font-sans text-[11px] uppercase tracking-wide text-muted">
              Area trend
            </h2>
            <LakeTrend lake={lake} height={190} />
          </div>

          <div>
            <h2 className="mb-1 font-sans text-[11px] uppercase tracking-wide text-muted">
              Per-year observations
            </h2>
            <table className="w-full border border-line font-sans text-[11px]">
              <thead>
                <tr className="bg-canvas text-left text-muted">
                  <th className="px-2 py-1 font-normal">Year</th>
                  <th className="px-2 py-1 text-right font-normal">Area (ha)</th>
                  <th className="px-2 py-1 text-right font-normal">% of window</th>
                  <th className="px-2 py-1 text-right font-normal">No-data</th>
                  <th className="px-2 py-1 font-normal">Status</th>
                </tr>
              </thead>
              <tbody>
                {years.map(([y, v]) => (
                  <tr key={y} className="border-t border-line">
                    <td className="px-2 py-1 text-ink">{y}</td>
                    <td className="px-2 py-1 text-right text-ink">{v.area_ha?.toFixed(1)}</td>
                    <td className="px-2 py-1 text-right"
                      style={{ color: (v.patch_frac ?? 0) > 0.5 ? GROW : '#6b7280' }}>
                      {((v.patch_frac ?? 0) * 100).toFixed(0)}%
                    </td>
                    <td className="px-2 py-1 text-right text-muted">
                      {((v.nodata_frac ?? 0) * 100).toFixed(0)}%
                    </td>
                    <td className="px-2 py-1"
                      style={{ color: v.flag === 'ok' ? '#6b7280' : GROW }}>
                      {v.flag}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {(excluded.length > 0 || nearSaturation.length > 0 || lake.suspect) && (
            <div className="border border-dashed px-3 py-2 font-sans text-[11px] leading-relaxed"
              style={{ borderColor: GROW, color: '#7a2e1f' }}>
              <strong>Read with care.</strong>
              <ul className="mt-1 list-disc pl-4">
                {lake.suspect && (
                  <li>First-year area disagrees badly with the Hi-MAG 2017 baseline
                    ({lake.himag_2017_ha} ha) — segmentation may be unstable here.</li>
                )}
                {excluded.length > 0 && (
                  <li>{excluded.length} year(s) excluded from the trend:{' '}
                    {excluded.map(([y, v]) => `${y} (${v.flag})`).join(', ')}.</li>
                )}
                {nearSaturation.length > 0 && (
                  <li>The lake fills over half the analysis window in{' '}
                    {nearSaturation.map(([y]) => y).join(', ')} — those areas are closer to a
                    lower bound than a measurement, and the trend may be understated.</li>
                )}
              </ul>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
