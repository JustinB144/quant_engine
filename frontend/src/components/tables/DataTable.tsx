import React, { useState } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import { useVirtualizer } from '@tanstack/react-virtual'
import { ArrowUp, ArrowDown } from 'lucide-react'

interface DataTableProps<T> {
  data: T[]
  columns: ColumnDef<T, unknown>[]
  pageSize?: number
  enableVirtualization?: boolean
  globalFilter?: string
}

export default function DataTable<T>({
  data,
  columns,
  pageSize = 25,
  enableVirtualization,
  globalFilter,
}: DataTableProps<T>) {
  const [sorting, setSorting] = useState<SortingState>([])
  const shouldVirtualize = enableVirtualization ?? data.length > 100

  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    ...(!shouldVirtualize ? { getPaginationRowModel: getPaginationRowModel() } : {}),
    ...(pageSize && !shouldVirtualize ? { initialState: { pagination: { pageSize } } } : {}),
  })

  const parentRef = React.useRef<HTMLDivElement>(null)
  const rows = table.getRowModel().rows

  // Always call useVirtualizer to satisfy React's Rules of Hooks
  const virtualizer = useVirtualizer({
    count: shouldVirtualize ? rows.length : 0,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 32,
    overscan: 10,
  })

  return (
    <div>
      <div
        ref={parentRef}
        style={{
          overflowX: 'auto',
          overflowY: shouldVirtualize ? 'auto' : 'visible',
          maxHeight: shouldVirtualize ? 600 : undefined,
        }}
      >
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((header) => (
                  <th
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    style={{
                      backgroundColor: '#1c2028',
                      color: 'var(--text-primary)',
                      fontWeight: 600,
                      fontSize: 11,
                      fontFamily: 'Menlo, monospace',
                      borderBottom: '1px solid var(--border)',
                      padding: '6px 10px',
                      textAlign: 'left',
                      cursor: header.column.getCanSort() ? 'pointer' : 'default',
                      whiteSpace: 'nowrap',
                      position: 'sticky',
                      top: 0,
                      zIndex: 1,
                    }}
                  >
                    <span className="flex items-center gap-1">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() === 'asc' && <ArrowUp size={10} />}
                      {header.column.getIsSorted() === 'desc' && <ArrowDown size={10} />}
                    </span>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {shouldVirtualize ? (
              <>
                {virtualizer.getVirtualItems().length > 0 && (
                  <tr>
                    <td
                      style={{ height: virtualizer.getVirtualItems()[0].start, padding: 0 }}
                      colSpan={columns.length}
                    />
                  </tr>
                )}
                {virtualizer.getVirtualItems().map((vRow) => {
                  const row = rows[vRow.index]
                  return (
                    <tr key={row.id}>
                      {row.getVisibleCells().map((cell) => (
                        <td
                          key={cell.id}
                          style={{
                            backgroundColor: 'var(--bg-secondary)',
                            color: 'var(--text-secondary)',
                            fontSize: 11,
                            fontFamily: 'Menlo, monospace',
                            border: '1px solid #21262d',
                            padding: '6px 10px',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </td>
                      ))}
                    </tr>
                  )
                })}
                {virtualizer.getVirtualItems().length > 0 && (
                  <tr>
                    <td
                      style={{
                        height:
                          virtualizer.getTotalSize() -
                          (virtualizer.getVirtualItems().at(-1)?.end ?? 0),
                        padding: 0,
                      }}
                      colSpan={columns.length}
                    />
                  </tr>
                )}
              </>
            ) : (
              rows.map((row) => (
                <tr key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      style={{
                        backgroundColor: 'var(--bg-secondary)',
                        color: 'var(--text-secondary)',
                        fontSize: 11,
                        fontFamily: 'Menlo, monospace',
                        border: '1px solid #21262d',
                        padding: '6px 10px',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination (non-virtual only) */}
      {!shouldVirtualize && (
        <div className="flex items-center justify-between mt-2 px-1">
          <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
            {data.length} rows | Page {table.getState().pagination.pageIndex + 1} of{' '}
            {table.getPageCount()}
          </span>
          <div className="flex gap-1">
            <button
              onClick={() => table.previousPage()}
              disabled={!table.getCanPreviousPage()}
              className="px-2 py-1 rounded text-xs"
              style={{
                backgroundColor: 'var(--bg-tertiary)',
                border: '1px solid var(--border-light)',
                color: 'var(--text-secondary)',
                cursor: table.getCanPreviousPage() ? 'pointer' : 'default',
                opacity: table.getCanPreviousPage() ? 1 : 0.4,
              }}
            >
              Prev
            </button>
            <button
              onClick={() => table.nextPage()}
              disabled={!table.getCanNextPage()}
              className="px-2 py-1 rounded text-xs"
              style={{
                backgroundColor: 'var(--bg-tertiary)',
                border: '1px solid var(--border-light)',
                color: 'var(--text-secondary)',
                cursor: table.getCanNextPage() ? 'pointer' : 'default',
                opacity: table.getCanNextPage() ? 1 : 0.4,
              }}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
