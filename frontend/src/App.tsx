import { useEffect, useState } from "react";
import { errorMessage } from "./api-errors";
import { api } from "@api";
import { Button, PageLoader } from "./components";
import { WarningIcon } from "./icons";
import { useRoute } from "./lib/routing";
import { ApplyPage } from "./pages/ApplyPage";
import { PreviewPage } from "./pages/PreviewPage";
import { SourcePage } from "./pages/SourcePage";
import type { Bootstrap } from "./types";

export function App() {
  const route = useRoute();
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadBootstrap = async () => {
      try {
        setBootstrap(await api.bootstrap());
      } catch (requestError: unknown) {
        setError(errorMessage(requestError));
      }
    };
    void loadBootstrap();
  }, []);

  if (error) {
    return (
      <div className="fatal-state">
        <WarningIcon size={28} />
        <h1>Folionym could not start</h1>
        <p>{error}</p>
        <Button onClick={() => {
          window.location.reload();
        }}>Try again</Button>
      </div>
    );
  }
  if (!bootstrap) return <PageLoader />;
  if (route === "preview") return <PreviewPage bootstrap={bootstrap} />;
  if (route === "apply") return <ApplyPage bootstrap={bootstrap} />;
  return <SourcePage bootstrap={bootstrap} onBootstrapChange={setBootstrap} />;
}
