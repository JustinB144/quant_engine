import React, { useState, useEffect } from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import DataTable from '@/components/tables/DataTable'
import JobMonitor from '@/components/job/JobMonitor'
import { useTrainModel } from '@/api/mutations/useTrainModel'
import { useModelVersions } from '@/api/queries/useModels'
import { useConfigStatus } from '@/api/queries/useConfig'
import { createColumnHelper } from '@tanstack/react-table'
import { Play } from 'lucide-react'
import type { TrainRequest } from '@/types/compute'
import type { ModelVersionInfo } from '@/types/models'

const FALLBACK_DEFAULTS: TrainRequest = {
  horizons: [5, 10, 20],
  feature_mode: 'full',
  survivorship_filter: true,
  full_universe: false,
}

const col = createColumnHelper<ModelVersionInfo>()
const columns = [
  col.accessor('version_id', { header: 'Version', cell: (info) => info.getValue().slice(0, 12) }),
  col.accessor('training_date', { header: 'Trained' }),
  col.accessor('cv_gap', { header: 'CV Gap', cell: (info) => info.getValue().toFixed(4) }),
  col.accessor('holdout_r2', { header: 'R\u00b2', cell: (info) => info.getValue().toFixed(4) }),
  col.accessor('holdout_spearman', { header: 'IC', cell: (info) => info.getValue().toFixed(4) }),
  col.accessor('n_features', { header: 'Features' }),
  col.accessor('n_samples', { header: 'Samples' }),
]

export default function TrainingTab() {
  const trainModel = useTrainModel()
  const { data: versionsData, isLoading } = useModelVersions()
  const { data: statusData } = useConfigStatus()
  const versions = versionsData?.data ?? []
  const [jobId, setJobId] = useState<string | null>(null)
  const [config, setConfig] = useState<TrainRequest>(FALLBACK_DEFAULTS)
  const [usingServerDefaults, setUsingServerDefaults] = useState(false)

  useEffect(() => {
    if (!statusData?.data) return
    const training = statusData.data.training
    const autopilot = statusData.data.autopilot
    if (!training && !autopilot) return

    const horizons = training?.forward_horizons?.value
    const featureMode = (training?.feature_mode?.value ?? autopilot?.feature_mode?.value) as string | undefined

    setConfig({
      horizons: Array.isArray(horizons) ? horizons : FALLBACK_DEFAULTS.horizons,
      feature_mode: typeof featureMode === 'string' ? featureMode : FALLBACK_DEFAULTS.feature_mode,
      survivorship_filter: true,
      full_universe: false,
    })
    setUsingServerDefaults(true)
  }, [statusData])

  const handleTrain = () => {
    trainModel.mutate(config, {
      onSuccess: (data) => {
        const id = (data.data as { job_id?: string })?.job_id
        if (id) setJobId(id)
      },
    })
  }

  return (
    <div>
      <div className="card-panel mb-4">
        <div className="card-panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          Training Configuration
          {usingServerDefaults && (
            <span className="font-mono" style={{ fontSize: 10, color: 'var(--accent-green)', opacity: 0.7 }}>
              Using server defaults
            </span>
          )}
        </div>
        <div className="grid grid-cols-4 gap-3 mb-3">
          <div>
            <label className="font-mono block mb-1" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>Feature Mode</label>
            <select
              value={config.feature_mode}
              onChange={(e) => setConfig({ ...config, feature_mode: e.target.value })}
              className="w-full rounded px-2 py-1.5 font-mono"
              style={{ fontSize: 11, backgroundColor: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              <option value="full">Full</option>
              <option value="reduced">Reduced</option>
            </select>
          </div>
          <div className="flex items-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={config.survivorship_filter} onChange={(e) => setConfig({ ...config, survivorship_filter: e.target.checked })} />
              <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Survivorship Filter</span>
            </label>
          </div>
          <div className="flex items-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={config.full_universe} onChange={(e) => setConfig({ ...config, full_universe: e.target.checked })} />
              <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Full Universe</span>
            </label>
          </div>
          <div className="flex items-end">
            <button
              onClick={handleTrain}
              disabled={trainModel.isPending}
              className="flex items-center gap-1.5 px-4 py-2 rounded-md w-full justify-center"
              style={{ backgroundColor: 'var(--accent-blue)', color: 'var(--text-primary)', fontSize: 12, cursor: trainModel.isPending ? 'wait' : 'pointer', border: 'none' }}
            >
              <Play size={12} /> Train Model
            </button>
          </div>
        </div>
        {jobId && <JobMonitor jobId={jobId} />}
      </div>

      <ChartContainer title="Model Version History" isLoading={isLoading} isEmpty={versions.length === 0} emptyMessage="No model versions found">
        <DataTable data={versions} columns={columns} pageSize={15} />
      </ChartContainer>
    </div>
  )
}
