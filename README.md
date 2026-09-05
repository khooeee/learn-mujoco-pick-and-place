TwinPick — photo of a real table, robot arm in the browser, small pick policy on your Mac.

## Run the sim (no API keys)

```bash
npm install
npm run dev
```

Open http://localhost:5173

1. Drop a table photo (it textures the stand-in table).
2. Drop an object photo (it wraps the stand-in cup).
3. Click **Train pick policy** (a few seconds).
4. Click **New cup spot**, then **Pick it up**. Switch Expert IK vs Learned.

## Cloud 3D (World Labs, Tripo, mint) via Convex

```bash
npx convex dev
```

Log in, create a project. Then:

```bash
npx convex env set WORLD_LABS_API_KEY ...
npx convex env set TRIPO_API_KEY ...
npx convex env set MINT_API_KEY ...
```

Put the printed `VITE_CONVEX_URL` in `.env.local`. Restart Vite. The three generate buttons turn on.

## Optional PyTorch on the M5

```bash
pip install torch
python ml/train_pick.py
```

Writes `ml/policy.json` (same net layout as the browser trainer).
