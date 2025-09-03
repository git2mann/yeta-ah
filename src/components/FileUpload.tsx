import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText } from 'lucide-react';

interface FileUploadProps {
  onFileUpload: (content: string) => void;
  accept?: Record<string, string[]>;
  maxSize?: number;
}

const FileUpload: React.FC<FileUploadProps> = ({ 
  onFileUpload, 
  accept = { 'text/csv': ['.csv'] },
  maxSize = 5242880 // 5MB
}) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        onFileUpload(content);
      };
      reader.readAsText(file);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept,
    maxSize,
    multiple: false
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
        isDragActive
          ? 'border-primary-500 bg-primary-50'
          : isDragReject
          ? 'border-red-500 bg-red-50'
          : 'border-gray-300 hover:border-gray-400'
      }`}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center">
        {isDragActive ? (
          <Upload className="h-12 w-12 text-primary-500 mb-4" />
        ) : (
          <FileText className="h-12 w-12 text-gray-400 mb-4" />
        )}
        <p className="text-lg font-medium text-gray-900 mb-2">
          {isDragActive ? 'Drop the file here' : 'Upload CSV file'}
        </p>
        <p className="text-sm text-gray-600">
          Drag and drop your CSV file here, or click to select
        </p>
        <p className="text-xs text-gray-500 mt-2">
          Maximum file size: {(maxSize / 1024 / 1024).toFixed(1)}MB
        </p>
      </div>
    </div>
  );
};

export default FileUpload;