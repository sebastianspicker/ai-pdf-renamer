import { useEffect, useState } from "react";

export type Route = "source" | "preview" | "apply";

export function currentRoute(): Route {
  if (window.location.pathname.startsWith("/preview")) return "preview";
  if (window.location.pathname.startsWith("/apply")) return "apply";
  return "source";
}

export function navigate(route: Route) {
  const path = route === "source" ? "/" : `/${route}`;
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function useRoute() {
  const [route, setRoute] = useState<Route>(currentRoute());
  useEffect(() => {
    const update = () => setRoute(currentRoute());
    window.addEventListener("popstate", update);
    return () => window.removeEventListener("popstate", update);
  }, []);
  return route;
}
