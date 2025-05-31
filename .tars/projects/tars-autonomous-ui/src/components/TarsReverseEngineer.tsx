// TARS Reverse Engineering Interface - Autonomously created by TARS
// Provides UI for analyzing and improving existing codebases
// TARS_REVERSE_ENGINEERING_UI_SIGNATURE: AUTONOMOUS_CODEBASE_IMPROVEMENT_INTERFACE

import React, { useState, useCallback } from 'react';
import { Search, FileText, Zap, Shield, Wrench, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';
import { TarsReverseEngineer, CodebaseAnalysis, Improvement } from '../utils/reverseEngineering';

export const TarsReverseEngineer: React.FC = () => {
  const [analysis, setAnalysis] = useState<CodebaseAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedProject, setSelectedProject] = useState('');
  const [applyingImprovements, setApplyingImprovements] = useState(false);

  // TARS autonomous project analysis
  const analyzeProject = useCallback(async () => {
    if (!selectedProject) return;
    
    setIsAnalyzing(true);
    console.log('🔍 TARS: Starting autonomous codebase analysis...');
    
    try {
      const reverseEngineer = new TarsReverseEngineer();
      const result = await reverseEngineer.analyzeProject(selectedProject);
      setAnalysis(result);
      console.log('✅ TARS: Analysis complete!');
    } catch (error) {
      console.error('❌ TARS: Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedProject]);

  // TARS autonomous improvement application
  const applyImprovements = useCallback(async (improvements: Improvement[]) => {
    if (!analysis) return;
    
    setApplyingImprovements(true);
    console.log('🔧 TARS: Applying autonomous improvements...');
    
    try {
      const reverseEngineer = new TarsReverseEngineer();
      const results = await reverseEngineer.applyImprovements(improvements, selectedProject);
      console.log(`🎉 TARS: Applied ${results.filter(r => r.success).length} improvements`);
    } catch (error) {
      console.error('❌ TARS: Failed to apply improvements:', error);
    } finally {
      setApplyingImprovements(false);
    }
  }, [analysis, selectedProject]);

  return (
    <div className="space-y-6">
      {/* TARS Reverse Engineering Header */}
      <div className="flex items-center space-x-3">
        <Search className="h-8 w-8 text-cyan-400" />
        <h2 className="text-3xl font-bold text-cyan-400 font-mono">TARS Reverse Engineering</h2>
        <span className="text-gray-400">Autonomous Codebase Analysis & Improvement</span>
      </div>

      {/* Project Selection */}
      <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
          <FileText className="h-5 w-5 text-blue-400" />
          <span>Project Analysis</span>
        </h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-gray-400 text-sm mb-2">Project Path</label>
            <input
              type="text"
              value={selectedProject}
              onChange={(e) => setSelectedProject(e.target.value)}
              placeholder="/path/to/project or project-name"
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
            />
          </div>
          
          <button
            onClick={analyzeProject}
            disabled={!selectedProject || isAnalyzing}
            className="bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 text-white px-4 py-2 rounded flex items-center space-x-2"
          >
            <Search className="h-4 w-4" />
            <span>{isAnalyzing ? 'TARS Analyzing...' : 'Analyze Project'}</span>
          </button>
        </div>
      </div>

      {/* Analysis Results */}
      {analysis && (
        <div className="space-y-6">
          {/* Project Overview */}
          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <h3 className="text-xl font-bold text-white mb-4">Project Overview</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gray-700 p-4 rounded">
                <p className="text-gray-400 text-sm">Project Type</p>
                <p className="text-white font-bold">{analysis.projectInfo.type}</p>
              </div>
              <div className="bg-gray-700 p-4 rounded">
                <p className="text-gray-400 text-sm">Framework</p>
                <p className="text-white font-bold">{analysis.projectInfo.framework}</p>
              </div>
              <div className="bg-gray-700 p-4 rounded">
                <p className="text-gray-400 text-sm">Lines of Code</p>
                <p className="text-white font-bold">{analysis.projectInfo.size.linesOfCode.toLocaleString()}</p>
              </div>
              <div className="bg-gray-700 p-4 rounded">
                <p className="text-gray-400 text-sm">Files</p>
                <p className="text-white font-bold">{analysis.projectInfo.size.files}</p>
              </div>
            </div>
          </div>

          {/* Code Quality Metrics */}
          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-green-400" />
              <span>Code Quality Analysis</span>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-700 p-4 rounded">
                <p className="text-gray-400 text-sm">Maintainability Index</p>
                <div className="flex items-center space-x-2">
                  <div className="w-full bg-gray-600 rounded-full h-2">
                    <div 
                      className="bg-green-400 h-2 rounded-full" 
                      style={{ width: `${analysis.codeQuality.maintainability.score}%` }}
                    />
                  </div>
                  <span className="text-white font-bold">{analysis.codeQuality.maintainability.score}/100</span>
                </div>
              </div>
              <div className="bg-gray-700 p-4 rounded">
                <p className="text-gray-400 text-sm">Test Coverage</p>
                <div className="flex items-center space-x-2">
                  <div className="w-full bg-gray-600 rounded-full h-2">
                    <div 
                      className="bg-blue-400 h-2 rounded-full" 
                      style={{ width: `${analysis.codeQuality.testCoverage.percentage}%` }}
                    />
                  </div>
                  <span className="text-white font-bold">{analysis.codeQuality.testCoverage.percentage}%</span>
                </div>
              </div>
              <div className="bg-gray-700 p-4 rounded">
                <p className="text-gray-400 text-sm">Code Smells</p>
                <p className="text-white font-bold">{analysis.codeQuality.codeSmells.length}</p>
              </div>
            </div>
          </div>

          {/* Improvement Recommendations */}
          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                <Wrench className="h-5 w-5 text-yellow-400" />
                <span>TARS Improvement Recommendations</span>
              </h3>
              <button
                onClick={() => applyImprovements([
                  ...analysis.improvements.critical,
                  ...analysis.improvements.high,
                  ...analysis.improvements.quickWins
                ].filter(i => i.tarsCanFix))}
                disabled={applyingImprovements}
                className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white px-4 py-2 rounded flex items-center space-x-2"
              >
                <Zap className="h-4 w-4" />
                <span>{applyingImprovements ? 'Applying...' : 'Auto-Fix All'}</span>
              </button>
            </div>

            {/* Critical Issues */}
            {analysis.improvements.critical.length > 0 && (
              <div className="mb-6">
                <h4 className="text-lg font-bold text-red-400 mb-3 flex items-center space-x-2">
                  <AlertTriangle className="h-4 w-4" />
                  <span>Critical Issues ({analysis.improvements.critical.length})</span>
                </h4>
                <div className="space-y-3">
                  {analysis.improvements.critical.map((improvement) => (
                    <ImprovementCard key={improvement.id} improvement={improvement} />
                  ))}
                </div>
              </div>
            )}

            {/* High Priority */}
            {analysis.improvements.high.length > 0 && (
              <div className="mb-6">
                <h4 className="text-lg font-bold text-orange-400 mb-3">
                  High Priority ({analysis.improvements.high.length})
                </h4>
                <div className="space-y-3">
                  {analysis.improvements.high.slice(0, 5).map((improvement) => (
                    <ImprovementCard key={improvement.id} improvement={improvement} />
                  ))}
                </div>
              </div>
            )}

            {/* Quick Wins */}
            {analysis.improvements.quickWins.length > 0 && (
              <div>
                <h4 className="text-lg font-bold text-green-400 mb-3 flex items-center space-x-2">
                  <Zap className="h-4 w-4" />
                  <span>Quick Wins ({analysis.improvements.quickWins.length})</span>
                </h4>
                <div className="space-y-3">
                  {analysis.improvements.quickWins.slice(0, 5).map((improvement) => (
                    <ImprovementCard key={improvement.id} improvement={improvement} />
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Security Analysis */}
          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
              <Shield className="h-5 w-5 text-red-400" />
              <span>Security Analysis</span>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-700 p-4 rounded">
                <p className="text-gray-400 text-sm">Security Score</p>
                <div className="flex items-center space-x-2">
                  <div className="w-full bg-gray-600 rounded-full h-2">
                    <div 
                      className="bg-red-400 h-2 rounded-full" 
                      style={{ width: `${analysis.security.score}%` }}
                    />
                  </div>
                  <span className="text-white font-bold">{analysis.security.score}/100</span>
                </div>
              </div>
              <div className="bg-gray-700 p-4 rounded">
                <p className="text-gray-400 text-sm">Vulnerabilities</p>
                <p className="text-white font-bold">{analysis.security.vulnerabilities.length}</p>
              </div>
            </div>
          </div>

          {/* Modernization Plan */}
          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <h3 className="text-xl font-bold text-white mb-4">TARS Modernization Plan</h3>
            <div className="space-y-4">
              <div>
                <p className="text-gray-400 text-sm">Current State</p>
                <p className="text-white">{analysis.modernization.currentState}</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Target State</p>
                <p className="text-white">{analysis.modernization.targetState}</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Estimated Timeline</p>
                <p className="text-white">{analysis.modernization.timeline}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Improvement Card Component
const ImprovementCard: React.FC<{ improvement: Improvement }> = ({ improvement }) => {
  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'critical': return 'text-red-400';
      case 'high': return 'text-orange-400';
      case 'medium': return 'text-yellow-400';
      case 'low': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'performance': return <TrendingUp className="h-4 w-4" />;
      case 'security': return <Shield className="h-4 w-4" />;
      case 'maintainability': return <Wrench className="h-4 w-4" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  return (
    <div className="bg-gray-700 p-4 rounded border-l-4 border-cyan-400">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            {getCategoryIcon(improvement.category)}
            <h5 className="font-bold text-white">{improvement.title}</h5>
            <span className={`text-xs px-2 py-1 rounded ${getImpactColor(improvement.impact)}`}>
              {improvement.impact}
            </span>
            {improvement.tarsCanFix && (
              <span className="text-xs px-2 py-1 rounded bg-green-600 text-white">
                TARS Can Fix
              </span>
            )}
          </div>
          <p className="text-gray-300 text-sm mb-2">{improvement.description}</p>
          <div className="flex items-center space-x-4 text-xs text-gray-400">
            <span>Effort: {improvement.effort}</span>
            <span>Files: {improvement.files.length}</span>
            {improvement.automatable && <span className="text-green-400">Automatable</span>}
          </div>
        </div>
        {improvement.tarsCanFix && (
          <CheckCircle className="h-5 w-5 text-green-400 ml-4" />
        )}
      </div>
    </div>
  );
};
