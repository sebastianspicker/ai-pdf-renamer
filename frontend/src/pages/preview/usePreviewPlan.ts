import { usePlanLoader } from "./usePlanLoader";
import { usePreviewSelection } from "./usePreviewSelection";

export function usePreviewPlan(planId: string | null) {
  const loader = usePlanLoader(planId);
  const selection = usePreviewSelection(loader.plan, loader.selected, loader.setSelected, loader.activeId);

  return {
    ...loader,
    ...selection,
  };
}
