import { Skeleton } from "@/components/ui/skeleton";

export function DashboardSkeleton() {
  return (
    <main className="mx-auto min-h-screen w-full max-w-[1680px] space-y-5 px-4 py-5 sm:px-6 lg:px-8">
      <div className="flex items-center justify-between border-b border-border pb-5">
        <div className="space-y-3">
          <Skeleton className="h-3 w-36" />
          <Skeleton className="h-9 w-64" />
        </div>
        <Skeleton className="h-10 w-28" />
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        {Array.from({ length: 5 }, (_, index) => <Skeleton className="h-28" key={index} />)}
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
