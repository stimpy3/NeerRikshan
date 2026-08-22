import {
  Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

const GROW = '#b4442e'
const SHRINK = '#3d5c7a'

/**
 * Per-lake area time series.
 *
 * The Hi-MAG 2017 area is drawn as a reference line rather than as a data
 * point: it comes from a different sensor (30 m Landsat) and a different year,
 * so it is context for the model's series, not a member of it.
 */
export default function LakeTrend({ lake, height = 150 }) {
  if (!lake?.series) return null
  const data = Object.entries(lake.series)
    .map(([year, ha]) => ({ year: Number(year), ha }))
    .sort((a, b) => a.year - b.year)
  if (!data.length) return null

  const rising = (lake.slope_pct_per_yr ?? 0) > 0
  const colour = rising ? GROW : SHRINK
  const areas = data.map((d) => d.ha)
  const pad = Math.max((Math.max(...areas) - Math.min(...areas)) * 0.15, 0.5)

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
          <XAxis
            dataKey="year" tick={{ fontSize: 10, fill: '#6b7280' }}
            axisLine={{ stroke: '#d4d4d4' }} tickLine={false}
          />
          <YAxis
            width={38} tick={{ fontSize: 10, fill: '#6b7280' }}
            axisLine={false} tickLine={false}
            domain={[Math.min(...areas) - pad, Math.max(...areas) + pad]}
            tickFormatter={(v) => v.toFixed(0)}
          />
          <Tooltip
            contentStyle={{ fontSize: 11, borderRadius: 0, border: '1px solid #d4d4d4' }}
            formatter={(v) => [`${Number(v).toFixed(2)} ha`, 'area']}
          />
          {lake.himag_2017_ha != null && (
            <ReferenceLine
              y={lake.himag_2017_ha} stroke="#9aa3a8" strokeDasharray="3 3"
              label={{ value: 'Hi-MAG 2017', fontSize: 9, fill: '#9aa3a8', position: 'insideTopRight' }}
            />
          )}
          <Line
            type="monotone" dataKey="ha" stroke={colour} strokeWidth={2}
            dot={{ r: 2.5, fill: colour }} isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>

      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 font-sans text-[11px] text-muted">
        <span>
          trend{' '}
          <strong style={{ color: colour }}>
            {lake.slope_pct_per_yr >= 0 ? '+' : ''}
            {lake.slope_pct_per_yr?.toFixed(2)}%/yr
          </strong>
        </span>
        <span>R² {lake.r2?.toFixed(2)}</span>
        <span>{lake.n_years} yrs</span>
        {lake.r2 < 0.5 && <span className="text-[10px] uppercase tracking-wide">weak fit</span>}
        {lake.suspect && (
          <span className="text-[10px] uppercase tracking-wide" style={{ color: GROW }}>
            suspect baseline
          </span>
        )}
      </div>
    </div>
  )
}
