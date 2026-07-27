import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function IconBase({ size = 18, children, ...props }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
      {...props}
    >
      {children}
    </svg>
  );
}

export const ArrowRightIcon = (props: IconProps) => (
  <IconBase {...props}><path d="M5 12h14m-5-5 5 5-5 5" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" /></IconBase>
);
export const CheckIcon = (props: IconProps) => (
  <IconBase {...props}><path d="m6 12 4 4 8-9" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.9" /></IconBase>
);
export const ChevronIcon = (props: IconProps) => (
  <IconBase {...props}><path d="m9 6 6 6-6 6" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" /></IconBase>
);
export const CloseIcon = (props: IconProps) => (
  <IconBase {...props}><path d="m6 6 12 12M18 6 6 18" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" /></IconBase>
);
export const DocumentIcon = (props: IconProps) => (
  <IconBase {...props}><path d="M7 3h7l4 4v14H7zM14 3v5h4M9.5 13h6M9.5 16h6" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" /></IconBase>
);
export const FolderIcon = (props: IconProps) => (
  <IconBase {...props}><path d="M3 7.5h7l2-2h9v14H3z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.6" /></IconBase>
);
export const InfoIcon = (props: IconProps) => (
  <IconBase {...props}><circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" /><path d="M12 11v6m0-9.5v.1" stroke="currentColor" strokeLinecap="round" strokeWidth="2" /></IconBase>
);
export const MenuIcon = (props: IconProps) => (
  <IconBase {...props}><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" strokeLinecap="round" strokeWidth="1.7" /></IconBase>
);
export const RefreshIcon = (props: IconProps) => (
  <IconBase {...props}><path d="M19 8a8 8 0 1 0 1 6M19 4v4h-4" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.7" /></IconBase>
);
export const SearchIcon = (props: IconProps) => (
  <IconBase {...props}><circle cx="10.5" cy="10.5" r="6.5" stroke="currentColor" strokeWidth="1.7" /><path d="m15.5 15.5 4 4" stroke="currentColor" strokeLinecap="round" strokeWidth="1.7" /></IconBase>
);
export const ShieldIcon = (props: IconProps) => (
  <IconBase {...props}><path d="M12 3 5 6v5c0 4.8 2.8 8.1 7 10 4.2-1.9 7-5.2 7-10V6z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.6" /><path d="M12 8v5" stroke="currentColor" strokeLinecap="round" strokeWidth="1.6" /><circle cx="12" cy="16" fill="currentColor" r="1" /></IconBase>
);
export const SlidersIcon = (props: IconProps) => (
  <IconBase {...props}><path d="M4 7h10M18 7h2M4 17h2m4 0h10M9 4v6m0 4v6M17 4v6" stroke="currentColor" strokeLinecap="round" strokeWidth="1.6" /></IconBase>
);
export const WarningIcon = (props: IconProps) => (
  <IconBase {...props}><path d="m12 3 9 17H3z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.6" /><path d="M12 9v4m0 3v.1" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" /></IconBase>
);
