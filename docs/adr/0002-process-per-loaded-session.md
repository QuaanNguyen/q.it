# One worker process per loaded session

Reservations (pins and what-ifs) are capacity math only. A worker child starts only when a session is explicitly Loaded. Each loaded session gets its own process so a crash or Metal fault does not take down the daemon. Worker crashes mark Failed with no auto-respawn in v1; pins remain. After daemon restart, workers are not auto-started even if pins persist.
