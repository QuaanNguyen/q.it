import { useMemo, useRef } from "react";
import { fmtTokens } from "../api";

const SIDE = 16;
const CELLS = SIDE * SIDE;

type Cell = { row: number; col: number; rank: number };

function diagonalOrder(): Cell[] {
  const cells: Cell[] = [];
  for (let row = 0; row < SIDE; row++) {
    for (let col = 0; col < SIDE; col++) {
      cells.push({ row, col, rank: 0 });
    }
  }
  const ranked = [...cells].sort(
    (a, b) => a.row + a.col - (b.row + b.col) || a.row - b.row
  );
  ranked.forEach((cell, rank) => {
    cell.rank = rank;
  });
  return cells;
}

const ORDER = diagonalOrder();

export function ContextSquare({ used, total }: { used: number; total: number }) {
  const fraction = total > 0 ? Math.min(1, used / total) : 0;
  const filled = Math.round(fraction * CELLS);
  const previous = useRef(0);
  const from = previous.current;
  previous.current = filled;
  const heat = fraction >= 1 ? "full" : fraction >= 0.85 ? "hot" : "";
  const leftPct = Math.max(0, Math.round((1 - fraction) * 100));

  const cells = useMemo(
    () =>
      ORDER.map((cell) => {
        const on = cell.rank < filled;
        const newlyOn = on && cell.rank >= from;
        const delay = newlyOn ? Math.min(400, (cell.rank - from) * 6) : 0;
        return (
          <div
            key={cell.row * SIDE + cell.col}
            className={on ? "cell on" : "cell"}
            style={{ transitionDelay: `${delay}ms` }}
          />
        );
      }),
    [filled, from]
  );

  return (
    <div className="meter" aria-label="context window">
      <div className={`grid16 ${heat}`}>{cells}</div>
      <div className="reading">
        <strong>
          {fmtTokens(used)} / {fmtTokens(total)}
        </strong>
        {leftPct}% left
      </div>
    </div>
  );
}
