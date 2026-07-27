import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import { ChevronIcon, FolderIcon } from "../icons";
import type { Bootstrap, DirectoryListing } from "../types";
import { Button } from "./Button";
import { ErrorBanner } from "./ErrorBanner";
import { compactPath } from "./format";
import { Modal } from "./Modal";
import { TextInput } from "./TextInput";

export function FolderBrowser({
  bootstrap,
  open,
  initialPath,
  onClose,
  onChoose,
}: {
  bootstrap: Bootstrap;
  open: boolean;
  initialPath: string;
  onClose: () => void;
  onChoose: (path: string) => void;
}) {
  const [listing, setListing] = useState<DirectoryListing | null>(null);
  const [path, setPath] = useState(initialPath || bootstrap.roots[0]?.path || "/");
  const [error, setError] = useState("");
  const pathInputRef = useRef<HTMLInputElement>(null);
  const browseRequestRef = useRef(0);

  const browse = useCallback(async (nextPath: string) => {
    const requestId = ++browseRequestRef.current;
    setError("");
    try {
      const next = await api.filesystem(nextPath);
      if (requestId !== browseRequestRef.current) return;
      setListing(next);
      setPath(next.path);
    } catch (requestError) {
      if (requestId !== browseRequestRef.current) return;
      setError(errorMessage(requestError));
    }
  }, []);

  useEffect(() => {
    if (open) void browse(initialPath || bootstrap.roots[0]?.path || "/");
  }, [open, initialPath, bootstrap.roots, browse]);

  return (
    <Modal
      footer={
        <>
          <span className="modal__summary">{listing ? `${listing.pdf_count} PDFs in this folder` : ""}</span>
          <Button onClick={onClose}>Cancel</Button>
          <Button disabled={!listing} onClick={() => listing && onChoose(listing.path)} variant="primary">
            Choose folder
          </Button>
        </>
      }
      onClose={onClose}
      open={open}
      title="Choose a PDF folder"
      wide
    >
      <div className="path-entry">
        <TextInput
          aria-label="Folder path"
          onChange={(event) => {
            browseRequestRef.current += 1;
            setPath(event.target.value);
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") void browse(event.currentTarget.value);
          }}
          ref={pathInputRef}
          value={path}
        />
        <Button onClick={() => void browse(pathInputRef.current?.value ?? path)}>Go</Button>
      </div>
      {error && <ErrorBanner message={error} onDismiss={() => setError("")} />}
      <div className="folder-list">
        {listing?.parent && (
          <button className="folder-row" onClick={() => void browse(listing.parent!)}>
            <FolderIcon />
            <span>
              <strong>Parent folder</strong>
              <small>{compactPath(listing.parent)}</small>
            </span>
            <ChevronIcon />
          </button>
        )}
        {listing?.entries.map((entry) => (
          <button className="folder-row" key={entry.path} onClick={() => void browse(entry.path)}>
            <FolderIcon />
            <span>
              <strong>{entry.name}</strong>
              <small>{entry.pdf_count} direct PDFs</small>
            </span>
            <ChevronIcon />
          </button>
        ))}
        {listing && listing.entries.length === 0 && (
          <div className="empty-inline">
            <FolderIcon size={24} />
            <span>No subfolders here. You can still choose this folder.</span>
          </div>
        )}
      </div>
    </Modal>
  );
}
