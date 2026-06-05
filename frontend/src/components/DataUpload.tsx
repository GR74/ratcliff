import { useState } from "react";

import { postUpload } from "../lib/api";
import { UploadedData } from "../lib/types";

interface Props {
  onUploaded: (data: UploadedData) => void;
}

export function DataUpload({ onUploaded }: Props) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setUploading(true);
    setError(null);
    setFilename(file.name);
    try {
      const data = await postUpload(file);
      onUploaded(data);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="border-2 border-dashed border-slate-300 rounded p-6 bg-white">
      <div className="text-center space-y-2">
        <p className="text-sm text-slate-600">
          Drop a data file here, or
        </p>
        <label className="inline-block cursor-pointer px-3 py-1 rounded bg-accent text-white text-sm">
          Browse
          <input
            type="file"
            accept=".txt,.csv,text/plain,text/csv"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
            className="hidden"
          />
        </label>
        <p className="text-xs text-slate-500">
          twod3datanew format OR CSV with columns rt,cat,condition
        </p>
        {uploading && <p className="text-xs text-accent">Parsing {filename}...</p>}
        {error && <p className="text-xs text-red-600">{error}</p>}
      </div>
    </div>
  );
}
