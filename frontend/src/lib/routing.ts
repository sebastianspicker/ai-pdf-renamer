import { useEffect, useState } from "react";
import { isStaticDemo } from "./demo";

export type Route = "source" | "preview" | "apply";

const basePath = import.meta.env.BASE_URL.replace(/\/$/, "");

export function routePath(route: Route): string {
  const suffix = route === "source" ? "/" : `/${route}`;
  if (isStaticDemo) {
    return route === "source" ? `${basePath}/` : `${basePath}/#${suffix}`;
  }
  return `${basePath}${suffix}` || "/";
}

export function currentRoute(): Route {
  const pathname = isStaticDemo
    ? window.location.hash.replace(/^#/, "") || "/"
    : window.location.pathname.slice(basePath.length) || "/";
  if (pathname.startsWith("/preview")) return "preview";
  if (pathname.startsWith("/apply")) return "apply";
  return "source";
}

export function navigate(route: Route) {
  window.history.pushState({}, "", routePath(route));
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
