import { Skeleton } from "@/components/ui/skeleton";

export function DashboardSkeleton() {
  return (
    <main className="mx-auto min-h-screen w-full max-w-[1720px] space-y-4 px-2 py-2 sm:px-5 sm:py-4 lg:px-7">
      <div className="relative h-[640px] overflow-hidden border border-border bg-card sm:h-[520px] lg:h-[430px]">
        <div className="flex h-16 items-center justify-between border-b border-border px-4 sm:px-5">
          <div className="flex items-center gap-3">
            <Skeleton className="size-9" />
            <div className="space-y-2"><Skeleton className="h-2.5 w-36" /><Skeleton className="h-2 w-24" /></div>
          </div>
          <Skeleton className="h-8 w-24" />
        </div>
        <div className="grid gap-8 px-5 py-8 lg:grid-cols-[1fr_300px] lg:px-10">
          <div className="space-y-4"><Skeleton className="h-7 w-40" /><Skeleton className="h-16 w-full max-w-xl sm:h-24" /><Skeleton className="h-4 w-full max-w-lg" /></div>
          <Skeleton className="h-40" />
        </div>
        <div className="absolute inset-x-0 bottom-0 grid grid-cols-2 border-t border-border lg:grid-cols-5">{Array.from({ length: 5 }, (_, index) => <Skeleton className={index === 4 ? "col-span-2 h-24 rounded-none border-r border-border lg:col-span-1" : "h-24 rounded-none border-r border-border"} key={index} />)}</div>
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <Skeleton className="h-[620px]" />
        <div className="space-y-4">
          <Skeleton className="h-72" />
          <Skeleton className="h-80" />
        </div>
      </div>
    </main>
  );
}
