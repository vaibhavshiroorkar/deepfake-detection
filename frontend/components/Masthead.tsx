import Link from "next/link";
import { createClient } from "@/lib/supabase/server";
import NavMenu from "./NavMenu";

export default async function Masthead() {
  const supabase = createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  return (
    <header className="border-b border-rule">
      <div className="mx-auto max-w-5xl px-6 h-16 flex items-center justify-between">
        <Link href="/" className="font-display text-2xl tracking-tight">
          Veritas<span className="text-ember">.</span>
        </Link>
        <NavMenu email={user?.email ?? null} />
      </div>
    </header>
  );
}
