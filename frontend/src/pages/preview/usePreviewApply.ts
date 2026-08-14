import { useCallback, useState } from "react";
import { api, errorMessage } from "../../api";
import { useRun } from "../../hooks/useRun";
import { navigate } from "../../lib/routing";
import type { Plan, Run } from "../../types";

type UsePreviewApplyOptions = {
  plan: Plan | null;
  selected: Set<string>;
  setError: (error: string) => void;
};

export function usePreviewApply({ plan, selected, setError }: UsePreviewApplyOptions) {
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const onRunComplete = useCallback((run: Run) => {
    if (run.state === "completed" && run.report_id) {
      sessionStorage.setItem("folionym.report", run.report_id);
      navigate("apply");
    }
  }, []);
  const { run, error: runError, setError: setRunError } = useRun(runId, onRunComplete);

  const apply = async () => {
    if (!plan) return;
    try {
      const result = await api.apply(plan.id, plan.revision, [...selected]);
      setConfirmOpen(false);
      setRunId(result.run_id);
    } catch (requestError: unknown) {
      setError(errorMessage(requestError));
      setConfirmOpen(false);
    }
  };

  const cancelRun = () => {
    if (runId) void api.cancel(runId);
  };

  const closeRun = () => {
    setRunId(null);
    setRunError("");
  };

  return {
    apply,
    cancelRun,
    closeConfirm: () => setConfirmOpen(false),
    closeRun,
    confirmOpen,
    openConfirm: () => setConfirmOpen(true),
    run,
    runError,
  };
}
