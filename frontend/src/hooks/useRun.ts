import { useEffect, useState } from "react";
import { errorMessage } from "../api-errors";
import { api } from "@api";
import type { Run } from "../types";

export const terminalStates = new Set(["completed", "cancelled", "failed"]);

export function useRun(runId: string | null, onComplete: (run: Run) => void) {
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!runId) {
      setRun(null);
      return;
    }
    let active = true;
    let timeout = 0;
    const poll = async () => {
      try {
        const next = await api.run(runId);
        if (!active) return;
        setRun(next);
        if (terminalStates.has(next.state)) {
          onComplete(next);
          return;
        }
        timeout = window.setTimeout(poll, 350);
      } catch (requestError: unknown) {
        if (active) setError(errorMessage(requestError));
      }
    };
    void poll();
    return () => {
      active = false;
      window.clearTimeout(timeout);
    };
  }, [runId, onComplete]);
  return { run, error, setError };
}
