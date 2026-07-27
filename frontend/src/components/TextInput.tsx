import { type InputHTMLAttributes, forwardRef } from "react";

export const TextInput = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function TextInput(props, ref) {
    return <input className="input" ref={ref} {...props} />;
  },
);
