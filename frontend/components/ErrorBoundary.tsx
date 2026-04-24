"use client";

import { Component, type ReactNode } from "react";

type Props = {
  children: ReactNode;
  fallback?: ReactNode;
};

type State = {
  error: Error | null;
};

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: { componentStack: string }) {
    if (typeof console !== "undefined") {
      console.error("ErrorBoundary caught:", error, info.componentStack);
    }
  }

  reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="border border-alert bg-alert/5 px-5 py-4 text-sm">
          <div className="eyebrow text-alert mb-1">Something broke</div>
          <p className="text-ink leading-relaxed max-w-prose">
            This section failed to render. The rest of the page is fine.
            Refresh or try a different input.
          </p>
          <p className="mt-2 font-mono text-[11px] text-mute break-all">
            {this.state.error.message}
          </p>
          <button
            onClick={this.reset}
            className="mt-3 text-xs underline decoration-dotted text-ink hover:text-ember"
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
