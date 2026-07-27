import { type InputHTMLAttributes } from "react";

export function Checkbox(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input className="checkbox" type="checkbox" {...props} />;
}
