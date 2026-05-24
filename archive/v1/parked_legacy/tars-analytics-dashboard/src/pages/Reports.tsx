import React from 'react';
import { FileText, Download, Calendar, TrendingUp } from 'lucide-react';

function Reports() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Reports
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Generate and download comprehensive analytics reports.
          </p>
        </div>
        
        <button className="btn btn-primary flex items-center space-x-2">
          <Download className="w-4 h-4" />
          <span>Generate Report</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[
          { title: 'Monthly Analytics', icon: TrendingUp, description: 'Comprehensive monthly performance report' },
          { title: 'User Activity', icon: FileText, description: 'Detailed user engagement metrics' },
          { title: 'Revenue Report', icon: Calendar, description: 'Financial performance analysis' },
        ].map((report, index) => (
          <div key={index} className="card hover:shadow-md transition-shadow">
            <div className="card-content">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-primary-100 dark:bg-primary-900/20 rounded-lg">
                  <report.icon className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  {report.title}
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                {report.description}
              </p>
              <button className="btn btn-outline w-full">
                Generate
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Reports;
