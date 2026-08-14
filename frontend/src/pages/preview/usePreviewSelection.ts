import { useMemo, useState, type Dispatch, type SetStateAction } from "react";
import type { Plan, PreviewItem, PreviewStatus } from "../../types";

function isSelectable(item: PreviewItem): boolean {
  return item.status === "ready" || item.status === "review";
}

function matchesFilter(item: PreviewItem, filter: PreviewStatus | "all", normalizedQuery: string): boolean {
  if (filter !== "all" && item.status !== filter) return false;
  if (!normalizedQuery) return true;
  return (
    item.current_name.toLowerCase().includes(normalizedQuery) ||
    item.proposed_name?.toLowerCase().includes(normalizedQuery) === true
  );
}

function filterVisibleItems(plan: Plan | null, filter: PreviewStatus | "all", query: string): PreviewItem[] {
  if (!plan) return [];
  const normalizedQuery = query.trim().toLowerCase();
  return plan.items.filter((item) => matchesFilter(item, filter, normalizedQuery));
}

function toggleSelectedId(current: Set<string>, id: string): Set<string> {
  const next = new Set(current);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
}

function addSelectableItems(current: Set<string>, items: PreviewItem[]): Set<string> {
  const next = new Set(current);
  items.forEach((item) => next.add(item.id));
  return next;
}

export function usePreviewSelection(
  plan: Plan | null,
  selected: Set<string>,
  setSelected: Dispatch<SetStateAction<Set<string>>>,
  activeId: string | null,
) {
  const [filter, setFilter] = useState<PreviewStatus | "all">("all");
  const [query, setQuery] = useState("");

  const visibleItems = useMemo(() => filterVisibleItems(plan, filter, query), [filter, plan, query]);

  const selectable = visibleItems.filter(isSelectable);
  const activeItem = plan?.items.find((item) => item.id === activeId) ?? null;
  const allVisibleSelected = selectable.length > 0 && selectable.every((item) => selected.has(item.id));

  const toggle = (id: string) => {
    setSelected((current) => toggleSelectedId(current, id));
  };

  const selectVisible = () => {
    setSelected((current) => addSelectableItems(current, selectable));
  };

  return {
    activeId,
    activeItem,
    allVisibleSelected,
    filter,
    query,
    selectableCount: selectable.length,
    selected,
    setFilter,
    setQuery,
    setSelected,
    selectVisible,
    toggle,
    visibleItems,
  };
}
