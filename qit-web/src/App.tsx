import { useEffect, useState } from "react";
import { CapacityPage } from "./CapacityPage";
import { CatalogPage } from "./catalog/CatalogPage";
import { SettingsPage } from "./SettingsPage";

type Route = "catalog" | "capacity" | "settings";

function routeFromHash(): Route {
  const raw = location.hash.replace("#/", "");
  if (raw === "capacity" || raw === "settings") return raw;
  return "catalog";
}

export default function App() {
  const [route, setRoute] = useState<Route>(routeFromHash);

  useEffect(() => {
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    if (!location.hash) location.hash = "#/catalog";
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <>
      <aside>
        <div className="brand">
          q.it
          <small>local runtime</small>
        </div>
        <nav>
          <a href="#/catalog" className={route === "catalog" ? "active" : ""}>
            Catalog
          </a>
          <a href="#/capacity" className={route === "capacity" ? "active" : ""}>
            Capacity
          </a>
          <a href="#/settings" className={route === "settings" ? "active" : ""}>
            Settings
          </a>
        </nav>
      </aside>
      <main>
        {route === "catalog" && <CatalogPage />}
        {route === "capacity" && <CapacityPage />}
        {route === "settings" && <SettingsPage />}
      </main>
    </>
  );
}
