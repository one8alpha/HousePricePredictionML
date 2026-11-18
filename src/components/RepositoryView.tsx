import React, { useMemo, useState } from 'react';
import { Check, Copy, Download, FileText, Folder } from 'lucide-react';
import { generateRepositoryFiles, RepoFile } from '../ml/repoTemplates';
import { PipelineArtifacts } from '../ml/types';

export function RepositoryView({ artifacts }: { artifacts: PipelineArtifacts }) {
  const files = useMemo(() => generateRepositoryFiles(artifacts), [artifacts]);
  const [selectedPath, setSelectedPath] = useState<string>(files[3].path); // default to train_pipeline.py
  const [copied, setCopied] = useState(false);

  const activeFile: RepoFile =
    files.find((f) => f.path === selectedPath) ?? files[0];

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(activeFile.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // ignore clipboard permissions error
    }
  };

  const downloadFile = (file: RepoFile) => {
    const blob = new Blob([file.content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = file.filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const folders = Array.from(new Set(files.map((f) => f.folder)));

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="border-b border-slate-200 pb-5">
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 font-medium mb-1.5">
          <span>Stage 14</span>
          <span aria-hidden="true">·</span>
          <span>Model Deployment & GitHub Repository Structure</span>
          <span aria-hidden="true">·</span>
          <span>{files.length} Ready-to-Submit Files</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-900">
          Deployment Repository & Documentation Package
        </h1>
        <p className="mt-1.5 text-sm text-slate-600 max-w-3xl">
          Complete project repository organized exactly per the B.Tech CSE Individual ML Project submission tree, ready to push to GitHub.
        </p>
      </section>

      {/* Interactive Repository Explorer */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Directory Tree */}
        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="pb-3 border-b border-slate-200">
            <h2 className="text-sm font-semibold text-slate-900">
              Project/ Directory Tree
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Select any file to inspect or download
            </p>
          </div>

          <div className="space-y-3">
            {folders.map((folder) => {
              const folderFiles = files.filter((f) => f.folder === folder);
              return (
                <div key={folder} className="space-y-1">
                  {folder !== 'Root' && (
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-700 px-1.5 py-0.5">
                      <Folder className="w-3.5 h-3.5 text-slate-500" />
                      <span>{folder}/</span>
                    </div>
                  )}
                  <div className={folder !== 'Root' ? 'pl-4 space-y-1' : 'space-y-1'}>
                    {folderFiles.map((file) => {
                      const isActive = file.path === activeFile.path;
                      return (
                        <button
                          key={file.path}
                          type="button"
                          onClick={() => setSelectedPath(file.path)}
                          className={`w-full flex items-center justify-between gap-2 text-left px-2.5 py-1.5 rounded text-xs transition-colors ${
                            isActive
                              ? 'bg-blue-50 text-blue-800 font-medium border border-blue-200'
                              : 'text-slate-700 hover:bg-slate-100 border border-transparent'
                          }`}
                        >
                          <span className="inline-flex items-center gap-1.5 truncate font-mono">
                            <FileText className="w-3.5 h-3.5 shrink-0 opacity-70" />
                            <span className="truncate">{file.filename}</span>
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: File Preview & Actions */}
        <div className="lg:col-span-8 bg-white border border-slate-200 rounded-lg overflow-hidden">
          <div className="px-5 py-3.5 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3 bg-slate-50">
            <div>
              <div className="text-sm font-mono font-semibold text-slate-900">
                Project/{activeFile.path}
              </div>
              <div className="text-xs text-slate-500 mt-0.5">{activeFile.description}</div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={handleCopy}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-50 transition-colors whitespace-nowrap"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                {copied ? 'Copied' : 'Copy'}
              </button>
              <button
                type="button"
                onClick={() => downloadFile(activeFile)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-blue-700 hover:bg-blue-800 rounded transition-colors whitespace-nowrap"
              >
                <Download className="w-3.5 h-3.5" />
                Download File
              </button>
            </div>
          </div>

          <div className="p-5 bg-slate-950 text-slate-100 overflow-x-auto max-h-[540px] overflow-y-auto">
            <pre className="text-xs font-mono leading-relaxed whitespace-pre">
              {activeFile.content.length > 18000
                ? activeFile.content.slice(0, 18000) +
                  '\n\n... [Preview truncated for display — click "Download File" for complete CSV]'
                : activeFile.content}
            </pre>
          </div>
        </div>
      </section>

      {/* Submission & Viva Readiness Checklist */}
      <section className="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
        <div className="pb-3 border-b border-slate-200">
          <h2 className="text-base font-semibold text-slate-900">
            GitHub Submission & Deployment Instructions
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-600 leading-relaxed">
          <div className="space-y-2">
            <div className="font-semibold text-slate-900">Local Python & Streamlit Setup</div>
            <pre className="p-3 bg-slate-50 border border-slate-200 rounded font-mono text-[11px] text-slate-800 overflow-x-auto">
{`mkdir -p Project/{Dataset,Notebook,"Source Code","Trained Model","Front End","Project Documentation"}
pip install -r Requirements.txt
python "Source Code/train_pipeline.py"
streamlit run "Front End/app.py"`}
            </pre>
          </div>
          <div className="space-y-2">
            <div className="font-semibold text-slate-900">GitHub Repository Push</div>
            <pre className="p-3 bg-slate-50 border border-slate-200 rounded font-mono text-[11px] text-slate-800 overflow-x-auto">
{`git init
git add .
git commit -m "Roll No 1: Complete House Price Prediction ML Pipeline"
git branch -M main
git remote add origin https://github.com/<username>/house-price-ml.git
git push -u origin main`}
            </pre>
          </div>
        </div>
      </section>
    </div>
  );
}
