# Analyze in the dashboard

## Sub-features

- `qit dashboard --port 0` serves the built web app on loopback and prints the base URL.
- `GET /api/benchmarks` lists installed packs and cases.
- `GET /api/runs` lists history; `GET /api/runs/<id>` returns one run with every sample.
- The run detail shows output text, decision distributions, saved images, speech detail, and the MaxSim ranking with a raw run JSON download.
- The dashboard can start a run against an already available provider endpoint.

## How to get to it (user POV)

```sh
target/debug/qit dashboard --port 0
```

Open the printed base URL in a browser, select a run, and open a case row. The retrieval detail is the `MaxSim / N query vectors` disclosure containing the ranking and the raw JSON link.

## Driving it with the HTTP API

The drive starts the dashboard, waits for `q.it dashboard listening on <base>`, requests `<base>/api/runs/<id>`, and compares the sample `retrieval` object with the exported one. The web bundle is checked statically: `qit-runtime/web-dist/assets/index-*.js` must exist after `npm run build` in `qit-web`.

## Gotchas

- `--port 0` picks a free port; parse the listening line rather than assuming 3000.
- The dashboard serves the embedded `web-dist` bundle. Run `npm ci && npm run build` in `qit-web` after changing `qit-web/src`, or the UI shows the previous bundle.
- The API never recomputes scores; it returns stored samples, so an API mismatch points at storage or serialization, not at the provider.
