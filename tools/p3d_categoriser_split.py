"""Session categoriser variant that keeps incomplete reviews out of the main catalogue."""
from __future__ import annotations

from p3d_categoriser_session import SessionCategoriserApp
from p3d_categoriser_storage import incomplete_state_path, save_split_state


class SplitStateCategoriserApp(SessionCategoriserApp):
    """Persist completed and reviewed-incomplete classifications separately."""

    def save_state(self) -> None:
        resume_path = (
            self.current.model_path
            if self.current is not None
            else self._resume_model_path
        )
        complete_count, incomplete_count = save_split_state(
            self.output,
            categories=self.categories,
            state=self.state,
            failures=self.failures,
            resume_model_path=resume_path,
        )
        if hasattr(self, "status_var"):
            companion = incomplete_state_path(self.output)
            if incomplete_count:
                self.status_var.set(
                    f"Saved {complete_count} complete model(s) to {self.output}; "
                    f"{incomplete_count} incomplete review(s) to {companion}."
                )
            else:
                self.status_var.set(
                    f"Saved {complete_count} complete model(s) to {self.output}."
                )
