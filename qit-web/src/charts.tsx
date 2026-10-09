import { frontier, modelPalette, modelKey } from "./analysis";
import { providerLabel } from "./api";
import type { Run } from "./api";
import { createContext, useContext } from "react";
import type { ReactNode } from "react";

const ColorContext = createContext<Map<string, string>>(new Map());

export function ModelChartColors({
  runs,
  children,
}: {
  runs: Run[];
  children: ReactNode;
}) {
  const colors = modelPalette(runs.map((run) => run.model));
  return (
    <ColorContext.Provider value={colors}>{children}</ColorContext.Provider>
  );
}

function useModelColor() {
  const colors = useContext(ColorContext);
  return (model: string) => colors.get(model) ?? "#e7e6db";
}

type Metric = (run: Run) => number | null;
type Format = (value: number) => string;

function ChartPanel({
  title,
  number,
  detail,
  meta,
  children,
}: {
  title: string;
  number: string;
  detail: string;
  meta: string;
  children: ReactNode;
}) {
  return (
    <article className="panel chart-panel">
      <div className="chart-heading">
        <div>
          <p className="eyebrow">[ {number} / comparison ]</p>
          <h2>{title}</h2>
          <p>{detail}</p>
        </div>
        <span className="chart-meta">{meta}</span>
      </div>
      {children}
    </article>
  );
}

function NoMeasurement({ message }: { message: string }) {
  return (
    <div className="chart-empty">
      <span>[ no measurement ]</span>
      <p>{message}</p>
    </div>
  );
}

export function RankingChart({
  title,
  number,
  detail,
  runs,
  value,
  tail,
  format,
  unit,
  higher = false,
  max,
  empty = "No measurement is available for this selection.",
}: {
  title: string;
  number: string;
  detail: string;
  runs: Run[];
  value: Metric;
  tail?: Metric;
  format: Format;
  unit: string;
  higher?: boolean;
  max?: number;
  empty?: string;
}) {
  const color = useModelColor();
  const usable = runs
    .filter((run) => value(run) !== null)
    .sort(
      (left, right) =>
        (value(left)! - value(right)!) * (higher ? -1 : 1) ||
        left.model.localeCompare(right.model),
    );
  const ceiling =
    max ??
    Math.max(...usable.flatMap((run) => [value(run)!, tail?.(run) ?? 0]), 1);
  const rank = (run: Run) =>
    1 +
    usable.filter((other) =>
      higher ? value(other)! > value(run)! : value(other)! < value(run)!,
    ).length;
  return (
    <ChartPanel
      title={title}
      number={number}
      detail={detail}
      meta={`${usable.length}/${runs.length} MODELS`}
    >
      {usable.length ? (
        <>
          <div className="ranking-key">
            <span>RANK / MODEL</span>
            <span>{tail ? "p50 / p95" : unit}</span>
          </div>
          <div className="ranked-bars">
            {usable.map((run) => (
              <a
                className="ranked-row"
                href={`#/runs?run=${encodeURIComponent(run.id)}`}
                key={run.id}
                title={`${run.model} / ${providerLabel(run.provider)} / ${run.base_url} / ${run.summary.successful_sample_count} samples`}
              >
                <span className="rank-number">
                  {String(rank(run)).padStart(2, "0")}
                </span>
                <div className="rank-body">
                  <div className="rank-label">
                    <span
                      className="chart-model"
                      style={{ color: color(run.model) }}
                    >
                      {run.model}
                    </span>
                    <strong>
                      {format(value(run)!)}
                      {tail && tail(run) !== null && (
                        <small> / {format(tail(run)!)}</small>
                      )}
                    </strong>
                  </div>
                  <div
                    className="rank-track"
                    style={{ color: color(run.model) }}
                  >
                    {tail && tail(run) !== null && (
                      <span
                        className="rank-tail"
                        style={{
                          width: `${Math.max(0, Math.min(100, (tail(run)! / ceiling) * 100))}%`,
                        }}
                      />
                    )}
                    <span
                      className="rank-fill"
                      style={{
                        width: `${Math.max(0, Math.min(100, (value(run)! / ceiling) * 100))}%`,
                      }}
                    />
                  </div>
                  <div className="rank-context">
                    <span>{providerLabel(run.provider)}</span>
                    <span>{run.summary.successful_sample_count} samples</span>
                  </div>
                </div>
              </a>
            ))}
          </div>
          <div className="chart-scale">
            <span>0</span>
            <span>{format(ceiling)}</span>
          </div>
        </>
      ) : (
        <NoMeasurement message={empty} />
      )}
    </ChartPanel>
  );
}

function extent(values: number[], zero = false): [number, number] {
  const low = Math.min(...values);
  const high = Math.max(...values);
  const padding = Math.max((high - low) * 0.15, Math.abs(high) * 0.06, 0.01);
  return [zero ? 0 : Math.max(0, low - padding), high + padding];
}

const bounds = {
  width: 540,
  height: 240,
  left: 78,
  right: 24,
  top: 18,
  bottom: 48,
};

function Axes({
  xDomain,
  yDomain,
  xFormat,
  yFormat,
  xLabel,
}: {
  xDomain: [number, number];
  yDomain: [number, number];
  xFormat: Format;
  yFormat: Format;
  xLabel: string;
}) {
  const { width, height, left, right, top, bottom } = bounds;
  return (
    <>
      {[0, 0.5, 1].map((ratio) => (
        <g key={ratio}>
          <line
            className="grid-line"
            x1={left}
            x2={width - right}
            y1={top + ratio * (height - top - bottom)}
            y2={top + ratio * (height - top - bottom)}
          />
          <text
            className="tick-label"
            x={left - 12}
            textAnchor="end"
            y={top + ratio * (height - top - bottom) + 4}
          >
            {yFormat(yDomain[1] - ratio * (yDomain[1] - yDomain[0]))}
          </text>
        </g>
      ))}
      <line
        className="axis-line"
        x1={left}
        x2={width - right}
        y1={height - bottom}
        y2={height - bottom}
      />
      <line
        className="axis-line"
        x1={left}
        x2={left}
        y1={top}
        y2={height - bottom}
      />
      <text className="tick-label" x={left} y={height - bottom + 19}>
        {xFormat(xDomain[0])}
      </text>
      <text
        className="tick-label"
        textAnchor="end"
        x={width - right}
        y={height - bottom + 19}
      >
        {xFormat(xDomain[1])}
      </text>
      <text
        className="axis-label"
        x={(left + width - right) / 2}
        y={height - 3}
      >
        {xLabel}
      </text>
    </>
  );
}

function Legend({ runs }: { runs: Run[] }) {
  const color = useModelColor();
  return (
    <div className="legend">
      {runs.map((run) => (
        <span
          key={modelKey(run)}
          title={run.model}
          style={{ color: color(run.model) }}
        >
          <i style={{ color: color(run.model) }}>+</i>
          {run.model}
        </span>
      ))}
    </div>
  );
}

export function ScatterChart({
  title,
  number,
  detail,
  runs,
  xValue,
  yValue,
  xFormat,
  yFormat,
  xLabel,
  yLabel,
  higherY = false,
}: {
  title: string;
  number: string;
  detail: string;
  runs: Run[];
  xValue: Metric;
  yValue: Metric;
  xFormat: Format;
  yFormat: Format;
  xLabel: string;
  yLabel: string;
  higherY?: boolean;
}) {
  const color = useModelColor();
  const usable = runs.filter(
    (run) => xValue(run) !== null && yValue(run) !== null,
  );
  const xDomain = extent(
    usable.map((run) => xValue(run)!),
    true,
  );
  const yDomain: [number, number] = higherY
    ? [0, 1]
    : extent(usable.map((run) => yValue(run)!));
  const { width, height, left, right, top, bottom } = bounds;
  const x = (value: number) =>
    left +
    ((value - xDomain[0]) / (xDomain[1] - xDomain[0])) * (width - left - right);
  const y = (value: number) =>
    top +
    ((yDomain[1] - value) / (yDomain[1] - yDomain[0])) *
      (height - top - bottom);
  const optimal = frontier(usable, xValue, yValue, higherY);
  const frontierPoints = usable
    .filter((run) => optimal.has(run.id))
    .sort((a, b) => xValue(a)! - xValue(b)!);
  return (
    <ChartPanel
      title={title}
      number={number}
      detail={detail}
      meta={`${usable.length} MODELS`}
    >
      {usable.length ? (
        <>
          <div className="plot-caption">{yLabel} / + model · [ ] frontier</div>
          <svg
            className="scatterplot"
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label={`${title} by model`}
          >
            <Axes
              xDomain={xDomain}
              yDomain={yDomain}
              xFormat={xFormat}
              yFormat={yFormat}
              xLabel={xLabel}
            />
            {frontierPoints.length > 1 && (
              <polyline
                className="frontier-line"
                points={frontierPoints
                  .map((run) => `${x(xValue(run)!)},${y(yValue(run)!)}`)
                  .join(" ")}
              />
            )}
            {usable.map((run) => (
              <a
                key={run.id}
                href={`#/runs?run=${encodeURIComponent(run.id)}`}
                aria-label={`${run.model}: ${xFormat(xValue(run)!)}, ${yFormat(yValue(run)!)}`}
              >
                <g>
                  {optimal.has(run.id) && (
                    <rect
                      className="frontier-box"
                      x={x(xValue(run)!) - 9}
                      y={y(yValue(run)!) - 9}
                      width={18}
                      height={18}
                    />
                  )}
                  <text
                    className="plot-point"
                    x={x(xValue(run)!)}
                    y={y(yValue(run)!) + 5}
                    textAnchor="middle"
                    fill={color(run.model)}
                  >
                    +
                    <title>{`${run.model}: ${xFormat(xValue(run)!)}, ${yFormat(yValue(run)!)}`}</title>
                  </text>
                </g>
              </a>
            ))}
          </svg>
          <Legend runs={usable} />
        </>
      ) : (
        <NoMeasurement message="Both measurements are needed to show the model tradeoff." />
      )}
    </ChartPanel>
  );
}

export function TrendChart({ runs }: { runs: Run[] }) {
  const color = useModelColor();
  const usable = runs
    .filter((run) => run.summary.latency_ms_p50 !== null)
    .sort((a, b) => a.started_at_ms - b.started_at_ms);
  const models = new Map<string, Run[]>();
  for (const run of usable)
    models.set(modelKey(run), [...(models.get(modelKey(run)) ?? []), run]);
  const dates = usable.map((run) => run.started_at_ms);
  const minimumDate = Math.min(...dates);
  const maximumDate = Math.max(...dates);
  const datePad = Math.max((maximumDate - minimumDate) * 0.08, 60000);
  const xDomain: [number, number] = [
    minimumDate - datePad,
    maximumDate + datePad,
  ];
  const yDomain = extent(
    usable.map((run) => run.summary.latency_ms_p50!),
    true,
  );
  const { width, height, left, right, top, bottom } = bounds;
  const x = (value: number) =>
    left +
    ((value - xDomain[0]) / (xDomain[1] - xDomain[0])) * (width - left - right);
  const y = (value: number) =>
    top +
    ((yDomain[1] - value) / (yDomain[1] - yDomain[0])) *
      (height - top - bottom);
  const formatDate = (value: number) =>
    new Date(value).toLocaleString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  return (
    <ChartPanel
      title="Latency over time"
      number="09"
      detail="Track repeat runs · same benchmark and setup"
      meta={`${usable.length} RUNS`}
    >
      {usable.length ? (
        <>
          <div className="plot-caption">
            Median latency (ms) / + successful run
          </div>
          <svg
            className="scatterplot"
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label="Model latency history"
          >
            <Axes
              xDomain={xDomain}
              yDomain={yDomain}
              xFormat={formatDate}
              yFormat={(value) => `${value.toFixed(0)} ms`}
              xLabel="Run start time"
            />
            {[...models].map(([key, series]) => (
              <polyline
                key={key}
                fill="none"
                stroke={color(series[0].model)}
                strokeWidth={1.5}
                points={series
                  .map(
                    (run) =>
                      `${x(run.started_at_ms)},${y(run.summary.latency_ms_p50!)}`,
                  )
                  .join(" ")}
              />
            ))}
            {usable.map((run) => (
              <a key={run.id} href={`#/runs?run=${encodeURIComponent(run.id)}`}>
                <text
                  className="plot-point"
                  fill={color(run.model)}
                  x={x(run.started_at_ms)}
                  y={y(run.summary.latency_ms_p50!) + 5}
                  textAnchor="middle"
                >
                  +
                  <title>{`${run.model}: ${run.summary.latency_ms_p50!.toFixed(1)} ms / ${new Date(run.started_at_ms).toLocaleString()}`}</title>
                </text>
              </a>
            ))}
          </svg>
          <Legend runs={[...models.values()].map((series) => series[0])} />
        </>
      ) : (
        <NoMeasurement message="Repeat the same benchmark setup to track changes over time." />
      )}
    </ChartPanel>
  );
}
