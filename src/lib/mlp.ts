/** Tiny 3→32→3 MLP. Trained with SGD in the browser (and optionally PyTorch). */

export type MLP = {
  inSize: number;
  hidden: number;
  outSize: number;
  W1: number[];
  b1: number[];
  W2: number[];
  b2: number[];
};

function randn() {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function initMLP(inSize = 3, hidden = 32, outSize = 3): MLP {
  const scale1 = Math.sqrt(2 / inSize);
  const scale2 = Math.sqrt(2 / hidden);
  return {
    inSize,
    hidden,
    outSize,
    W1: Array.from({ length: hidden * inSize }, () => randn() * scale1),
    b1: Array(hidden).fill(0),
    W2: Array.from({ length: outSize * hidden }, () => randn() * scale2),
    b2: Array(outSize).fill(0),
  };
}

function tanh(x: number) {
  return Math.tanh(x);
}

export function forward(m: MLP, x: number[]): number[] {
  const h = new Array(m.hidden);
  for (let i = 0; i < m.hidden; i++) {
    let s = m.b1[i];
    for (let j = 0; j < m.inSize; j++) s += m.W1[i * m.inSize + j] * x[j];
    h[i] = tanh(s);
  }
  const y = new Array(m.outSize);
  for (let i = 0; i < m.outSize; i++) {
    let s = m.b2[i];
    for (let j = 0; j < m.hidden; j++) s += m.W2[i * m.hidden + j] * h[j];
    y[i] = s;
  }
  return y;
}

export function trainEpoch(m: MLP, xs: number[][], ys: number[][], lr: number): number {
  const n = xs.length;
  let loss = 0;
  const gW1 = new Array(m.W1.length).fill(0);
  const gb1 = new Array(m.hidden).fill(0);
  const gW2 = new Array(m.W2.length).fill(0);
  const gb2 = new Array(m.outSize).fill(0);

  for (let k = 0; k < n; k++) {
    const x = xs[k];
    const t = ys[k];
    const pre = new Array(m.hidden);
    const h = new Array(m.hidden);
    for (let i = 0; i < m.hidden; i++) {
      let s = m.b1[i];
      for (let j = 0; j < m.inSize; j++) s += m.W1[i * m.inSize + j] * x[j];
      pre[i] = s;
      h[i] = tanh(s);
    }
    const y = new Array(m.outSize);
    for (let i = 0; i < m.outSize; i++) {
      let s = m.b2[i];
      for (let j = 0; j < m.hidden; j++) s += m.W2[i * m.hidden + j] * h[j];
      y[i] = s;
    }

    const dy = new Array(m.outSize);
    for (let i = 0; i < m.outSize; i++) {
      dy[i] = (2 / m.outSize) * (y[i] - t[i]);
      loss += (y[i] - t[i]) ** 2;
    }

    const dh = new Array(m.hidden).fill(0);
    for (let i = 0; i < m.outSize; i++) {
      gb2[i] += dy[i];
      for (let j = 0; j < m.hidden; j++) {
        gW2[i * m.hidden + j] += dy[i] * h[j];
        dh[j] += m.W2[i * m.hidden + j] * dy[i];
      }
    }
    for (let i = 0; i < m.hidden; i++) {
      const dt = (1 - h[i] * h[i]) * dh[i];
      gb1[i] += dt;
      for (let j = 0; j < m.inSize; j++) gW1[i * m.inSize + j] += dt * x[j];
    }
  }

  const inv = 1 / n;
  loss = loss / (n * m.outSize);
  for (let i = 0; i < m.W1.length; i++) m.W1[i] -= lr * gW1[i] * inv;
  for (let i = 0; i < m.b1.length; i++) m.b1[i] -= lr * gb1[i] * inv;
  for (let i = 0; i < m.W2.length; i++) m.W2[i] -= lr * gW2[i] * inv;
  for (let i = 0; i < m.b2.length; i++) m.b2[i] -= lr * gb2[i] * inv;
  return loss;
}
