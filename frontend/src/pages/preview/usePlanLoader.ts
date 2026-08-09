import { useEffect, useState } from "react";
import { api, errorMessage } from "../../api";
import { navigate } from "../../lib/routing";
import type { Plan } from "../../types";

export function usePlanLoader(planId: string | null) {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    if (!planId) {
      navigate("source");
      return;
    }

    let mounted = true;
    const loadPlan = async () => {
      try {
        const next = await api.plan(planId);
        if (mounted) {
          setPlan(next);
          setSelected(new Set(next.items.filter((item) => item.included).map((item) => item.id)));
          setActiveId(next.items[0]?.id ?? null);
        }
      } catch (requestError: unknown) {
        if (mounted) setError(errorMessage(requestError));
      } finally {
        if (mounted) setLoading(false);
      }
    };
    void loadPlan();

    return () => {
      mounted = false;
    };
  }, [planId]);

  return {
    activeId,
    error,
    loading,
    plan,
    selected,
    setActiveId,
    setError,
    setSelected,
  };
}
