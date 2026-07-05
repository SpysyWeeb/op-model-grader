"""Simple Tkinter UI: sign in, pick drives, request uploads, grade.

Launched with `opgrader --ui`. Pure stdlib (tkinter) + the existing
pipeline; all network and grading work runs in background threads and the
widgets are only touched from the Tk main loop (via root.after).
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import __version__
from . import connect as C
from .config import get_t_follow, set_t_follow

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as e:  # pragma: no cover - depends on the host python
    raise SystemExit(
        "tkinter is not available in this Python. Install your distro's "
        "python3-tk / python3-tkinter package and try again."
    ) from e

POLL_JOB_MS = 500
POLL_UPLOAD_MS = 30_000
PARTIAL_OK_FRACTION = 0.8


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"op-model-grader {__version__}")

        self.jobs = C.JobManager()
        self.jwt: str | None = None
        self.dongles: list[dict] = []
        self.routes: list[dict] = []          # summarize_route dicts
        self.badges: dict[str, dict] = {}     # fullname -> files_badge dict
        self.pending_uploads: set[str] = set()
        self.local_paths: list[str] = []
        self._last_grade_args: tuple[list[str], list[str], dict] | None = None

        self._q: queue.Queue = queue.Queue()
        self._build_widgets()
        self._apply_source()
        root.after(50, self._pump)
        self._bg(self._check_auth, self._on_auth_checked)
        self._refresh_reports()
        self._refresh_profile()
        # Catch-all: absorb any other late reflow (theme/font metrics
        # settling, etc.) beyond the two async paths already re-fit above.
        root.after(200, self._fit_window)

    # ------------------------------------------------------------ threading
    # Worker threads never touch Tk directly: they enqueue callbacks that the
    # main loop drains every 50 ms (root.after from other threads is unsafe).

    def _post(self, fn, *args):
        self._q.put((fn, args))

    def _pump(self):
        try:
            while True:
                fn, args = self._q.get_nowait()
                try:
                    fn(*args)
                except Exception:  # noqa: BLE001 - a bad callback must not kill the pump
                    pass
        except queue.Empty:
            pass
        self.root.after(50, self._pump)

    def _bg(self, fn, on_done, *args):
        """Run fn(*args) in a thread; call on_done(result, error) on the UI thread."""

        def worker():
            try:
                result, error = fn(*args), None
            except Exception as e:  # noqa: BLE001 - marshal errors to the UI
                result, error = None, e
            self._post(on_done, result, error)

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------------------------------------------- widgets

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}
        self._pad = pad

        # log source toggle
        src = ttk.Frame(self.root)
        src.pack(fill="x", **pad)
        ttk.Label(src, text="Grade from:").pack(side="left", padx=(4, 8))
        self.source_var = tk.StringVar(value="connect")
        ttk.Radiobutton(src, text="comma connect (your uploaded drives)", value="connect",
                        variable=self.source_var, command=self._apply_source).pack(side="left")
        ttk.Radiobutton(src, text="local rlog folders", value="local",
                        variable=self.source_var, command=self._apply_source).pack(side="left", padx=12)

        # auth bar
        self.auth_frame = ttk.LabelFrame(self.root, text="comma account")
        self.auth_frame.pack(fill="x", **pad)
        self.auth_label = ttk.Label(self.auth_frame, text="checking sign-in…")
        self.auth_label.pack(side="left", padx=8, pady=6)
        self.token_entry = ttk.Entry(self.auth_frame, width=42, show="*")
        self.token_save = ttk.Button(self.auth_frame, text="Save token", command=self._save_token)
        self.signout_btn = ttk.Button(self.auth_frame, text="Sign out", command=self._sign_out)
        self.token_help = ttk.Button(
            self.auth_frame, text="Get a token (jwt.comma.ai)",
            command=lambda: webbrowser.open("https://jwt.comma.ai"),
        )

        # device + routes
        routes_frame = ttk.LabelFrame(self.root, text="drives on your device")
        routes_frame.pack(fill="both", expand=True, **pad)
        self.routes_frame = routes_frame
        top = ttk.Frame(routes_frame)
        top.pack(fill="x", padx=6, pady=4)
        ttk.Label(top, text="Device:").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(top, textvariable=self.device_var, state="readonly", width=40)
        self.device_box.pack(side="left", padx=6)
        self.device_box.bind("<<ComboboxSelected>>", lambda _e: self._load_routes())
        ttk.Button(top, text="Refresh", command=self._load_routes).pack(side="left", padx=4)
        self.routes_msg = ttk.Label(top, text="", foreground="gray")
        self.routes_msg.pack(side="left", padx=10)

        cols = ("started", "duration", "segments", "branch", "vehicle", "rlogs")
        self.tree = ttk.Treeview(routes_frame, columns=cols, show="headings", selectmode="extended")
        headings = {
            "started": ("Started", 150), "duration": ("Duration", 80),
            "segments": ("Segments", 80), "branch": ("Branch", 130),
            "vehicle": ("Vehicle", 130), "rlogs": ("rlogs", 190),
        }
        for c in cols:
            text, width = headings[c]
            self.tree.heading(c, text=text)
            self.tree.column(c, width=width, anchor="w")
        yscroll = ttk.Scrollbar(routes_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
        yscroll.pack(side="left", fill="y", pady=4)
        self.tree.tag_configure("ready", foreground="#0ca30c")
        self.tree.tag_configure("partial", foreground="#b8860b")
        self.tree.tag_configure("none", foreground="gray")

        act = ttk.Frame(self.root)
        act.pack(fill="x", **pad)
        self.act_frame = act
        self.upload_btn = ttk.Button(
            act, text="Request upload for selected", command=self._request_upload
        )
        self.upload_btn.pack(side="left")
        self.cell_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(act, text="upload over cell data", variable=self.cell_var).pack(
            side="left", padx=8
        )
        self.upload_msg = ttk.Label(act, text="", foreground="gray")
        self.upload_msg.pack(side="left", padx=10)

        # local folders
        local = ttk.LabelFrame(self.root, text="local rlog folders")
        local.pack(fill="x", **pad)
        self.local_frame = local
        lrow = ttk.Frame(local)
        lrow.pack(fill="x", padx=6, pady=4)
        ttk.Button(lrow, text="Add folder…", command=self._add_folder).pack(side="left")
        ttk.Button(lrow, text="Remove selected", command=self._remove_folder).pack(side="left", padx=6)
        ttk.Label(lrow, text="each folder should contain rlog files or segment dirs",
                  foreground="gray").pack(side="left", padx=10)
        self.paths_list = tk.Listbox(local, height=5)
        self.paths_list.pack(fill="x", padx=6, pady=(0, 6))

        # grade + progress
        grade = ttk.LabelFrame(self.root, text="grade")
        grade.pack(fill="x", **pad)
        self.grade_frame = grade
        tf = ttk.Frame(grade)
        tf.pack(fill="x", padx=6, pady=(4, 0))
        ttk.Label(tf, text="personality follow targets (s):").pack(side="left")
        self.tf_vars: dict[str, tk.StringVar] = {}
        targets = get_t_follow()
        for p in ("aggressive", "standard", "relaxed"):
            ttk.Label(tf, text=f" {p}").pack(side="left", padx=(8, 2))
            var = tk.StringVar(value=f"{targets[p]:g}")
            ttk.Entry(tf, textvariable=var, width=6).pack(side="left")
            self.tf_vars[p] = var
        ttk.Label(tf, text="  (fork-dependent; stock: 1.25 / 1.45 / 1.75)",
                  foreground="gray").pack(side="left")
        prow = ttk.Frame(grade)
        prow.pack(fill="x", padx=6, pady=(0, 4))
        self.profile_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            prow, text="use driver profile for baselines", variable=self.profile_var
        ).pack(side="left")
        ttk.Label(
            prow, text="  (pools your manual driving across routes for a steadier baseline)",
            foreground="gray",
        ).pack(side="left")
        grow = ttk.Frame(grade)
        grow.pack(fill="x", padx=6, pady=4)
        self.grade_btn = ttk.Button(
            grow, text="Grade selected drives", command=self._grade
        )
        self.grade_btn.pack(side="left")
        self.progress = ttk.Progressbar(grow, length=280, mode="determinate")
        self.progress.pack(side="left", padx=10)
        self.status = ttk.Label(grade, text="select drives above, then grade", foreground="gray")
        self.status.pack(anchor="w", padx=8, pady=(0, 6))

        # driver profile panel
        prof = ttk.LabelFrame(self.root, text="driver profile")
        prof.pack(fill="x", **pad)
        prow2 = ttk.Frame(prof)
        prow2.pack(fill="x", padx=6, pady=6)
        self.profile_label = ttk.Label(prow2, text="checking…", foreground="gray")
        self.profile_label.pack(side="left")
        ttk.Button(
            prow2, text="Delete driver profile", command=self._delete_profile
        ).pack(side="right")

        # past reports
        rep = ttk.LabelFrame(self.root, text="past reports (double-click to open)")
        rep.pack(fill="x", **pad)
        self.reports_list = tk.Listbox(rep, height=3, selectmode="extended")
        self.reports_list.pack(fill="x", padx=6, pady=(6, 0))
        self.reports_list.bind("<Double-Button-1>", self._open_report)
        self._report_paths: list[str] = []
        rrow = ttk.Frame(rep)
        rrow.pack(fill="x", padx=6, pady=(2, 6))
        ttk.Button(rrow, text="Delete selected", command=self._delete_reports).pack(side="left")
        self.cache_label = ttk.Label(rrow, text="", foreground="gray")
        self.cache_label.pack(side="right", padx=(0, 4))
        ttk.Button(rrow, text="Clear downloaded rlogs", command=self._clear_cache).pack(side="right", padx=6)

    # --------------------------------------------------------------- source

    def _apply_source(self):
        """Show either the connect panels or the local-folder panel, then refit."""
        connect = self.source_var.get() == "connect"
        for f in (self.auth_frame, self.routes_frame, self.act_frame, self.local_frame):
            f.pack_forget()
        if connect:
            self.auth_frame.pack(fill="x", before=self.grade_frame, **self._pad)
            self.routes_frame.pack(fill="both", expand=True, before=self.grade_frame, **self._pad)
            self.act_frame.pack(fill="x", before=self.grade_frame, **self._pad)
        else:
            self.local_frame.pack(fill="x", before=self.grade_frame, **self._pad)
        self._fit_window()

    # Small cushion against platform/theme border-metric differences (window
    # manager decorations, ttk theme padding) that reqwidth/reqheight don't
    # always fully account for -- cheap insurance against a content row
    # landing a few pixels short of visible.
    _FIT_MARGIN = 16

    def _fit_window(self):
        """Size the window so the whole UI fits without manual resizing."""
        self.root.update_idletasks()
        w = max(880, self.root.winfo_reqwidth() + self._FIT_MARGIN)
        h = self.root.winfo_reqheight() + self._FIT_MARGIN
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(w, h)

    # ----------------------------------------------------------------- auth

    def _check_auth(self):
        self.jwt = C.read_jwt()
        if not self.jwt:
            return None
        try:
            return C.get_me(self.jwt)
        except C.ApiError:
            self.jwt = None
            return None

    def _on_auth_checked(self, me, err):
        if me:
            who = me.get("email") or me.get("user_id") or "signed in"
            self.auth_label.config(text=f"signed in as {who}")
            self.token_entry.pack_forget()
            self.token_save.pack_forget()
            self.token_help.pack_forget()
            self.signout_btn.pack(side="right", padx=6, pady=4)
            self._load_devices()
        else:
            self.auth_label.config(
                text="paste a comma JWT to browse your drives (local folders work without one):"
            )
            self.signout_btn.pack_forget()
            self.token_help.pack(side="right", padx=6, pady=4)
            self.token_save.pack(side="right", padx=6, pady=4)
            self.token_entry.pack(side="right", padx=6, pady=4, fill="x", expand=True)
        self._fit_window()  # auth-row content just changed shape; re-settle

    def _sign_out(self):
        if not messagebox.askyesno(
            "Sign out",
            "Remove the saved token from ~/.comma/auth.json?\n"
            "You'll need a fresh token from jwt.comma.ai to sign back in.",
        ):
            return
        C.clear_jwt()
        self.jwt = None
        self.dongles = []
        self.routes = []
        self.device_box["values"] = []
        self.device_var.set("")
        self.tree.delete(*self.tree.get_children())
        self.routes_msg.config(text="")
        self.token_entry.delete(0, "end")
        self._on_auth_checked(None, None)

    def _save_token(self):
        token = self.token_entry.get().strip()

        def save():
            C.get_me(token)  # validate before writing
            C.save_jwt(token)
            return True

        def done(_ok, err):
            if err:
                messagebox.showerror("token rejected", str(err))
            else:
                self._bg(self._check_auth, self._on_auth_checked)

        self._bg(save, done)

    # --------------------------------------------------------------- routes

    def _load_devices(self):
        def done(devices, err):
            if err:
                self.routes_msg.config(text=str(err))
                return
            self.dongles = devices or []
            names = [
                f"{d.get('alias') or d.get('device_type') or 'device'} — {d['dongle_id']}"
                for d in self.dongles
            ]
            self.device_box["values"] = names
            if names:
                self.device_box.current(0)
                self._load_routes()
            else:
                self.routes_msg.config(text="no devices on this account")

        self._bg(lambda: C.get_devices(self.jwt), done)

    def _current_dongle(self) -> str | None:
        i = self.device_box.current()
        return self.dongles[i]["dongle_id"] if 0 <= i < len(self.dongles) else None

    def _load_routes(self):
        dongle = self._current_dongle()
        if not dongle or not self.jwt:
            return
        self.routes_msg.config(text="loading…")

        def done(raw, err):
            if err:
                self.routes_msg.config(text=str(err))
                return
            self.routes = sorted(
                (C.summarize_route(r) for r in raw or []),
                key=lambda r: -(r["start_utc_millis"] or 0),
            )
            self.badges = {}
            self._render_routes()
            self.routes_msg.config(text=f"{len(self.routes)} drives")
            self._fetch_badges()

        self._bg(lambda: C.get_routes(dongle, self.jwt), done)

    def _render_routes(self):
        selected = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        for r in self.routes:
            b = self.badges.get(r["fullname"])
            if r["fullname"] in self.pending_uploads and (not b or b["kind"] != "ready"):
                badge_text, tag = "upload queued (uploads on WiFi)", "partial"
            elif b:
                badge_text, tag = b["label"], b["kind"]
            else:
                badge_text, tag = "checking…", "none"
            started = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(r["start_utc_millis"] / 1000))
                if r["start_utc_millis"] else "–"
            )
            dur = f"{r['duration_s'] / 60:.0f} min" if r["duration_s"] else "–"
            self.tree.insert(
                "", "end", iid=r["fullname"], tags=(tag,),
                values=(started, dur, r["n_segments"] or "–", r["git_branch"],
                        r["platform"] or "–", badge_text),
            )
        for iid in selected:
            if self.tree.exists(iid):
                self.tree.selection_add(iid)

    def _fetch_badges(self):
        routes = list(self.routes)

        def fetch():
            for r in routes:
                try:
                    files = C.get_route_files(r["fullname"], self.jwt)
                    badge = C.files_badge(files, r["n_segments"])
                except Exception:  # noqa: BLE001 - keep going per-route
                    badge = {"label": "unknown", "kind": "none", "n_logs": 0,
                             "n_segments": r["n_segments"]}
                self._post(self._set_badge, r["fullname"], badge)
            return True

        threading.Thread(target=fetch, daemon=True).start()

    def _set_badge(self, fullname: str, badge: dict):
        self.badges[fullname] = badge
        if badge["kind"] == "ready":
            self.pending_uploads.discard(fullname)
        self._render_routes()

    # -------------------------------------------------------------- uploads

    def _request_upload(self):
        sel = self.tree.selection()
        dongle = self._current_dongle()
        if not sel or not dongle:
            self.upload_msg.config(text="select a drive first")
            return
        allow_cell = self.cell_var.get()
        targets = [r for r in self.routes if r["fullname"] in sel]
        self.upload_msg.config(text="requesting…")

        def work():
            msgs = []
            for r in targets:
                try:
                    res = C.request_upload(
                        dongle, r["fullname"], r["n_segments"], self.jwt, allow_cell
                    )
                    self.pending_uploads.add(r["fullname"])
                    msgs.append(f"{r['name']}: {res['message']}")
                except C.ApiError as e:
                    msgs.append(f"{r['name']}: {e}")
            return msgs

        def done(msgs, err):
            text = str(err) if err else "; ".join(msgs or [])
            self.upload_msg.config(text=text[:140])
            self._render_routes()
            if self.pending_uploads:
                self.root.after(POLL_UPLOAD_MS, self._poll_pending_uploads)

        self._bg(work, done)

    def _poll_pending_uploads(self):
        if not self.pending_uploads or not self.jwt:
            return
        pending = [r for r in self.routes if r["fullname"] in self.pending_uploads]

        def fetch():
            for r in pending:
                try:
                    files = C.get_route_files(r["fullname"], self.jwt)
                    badge = C.files_badge(files, r["n_segments"])
                    self._post(self._set_badge, r["fullname"], badge)
                except Exception:  # noqa: BLE001
                    pass
            return True

        def done(_ok, _err):
            if self.pending_uploads:
                self.root.after(POLL_UPLOAD_MS, self._poll_pending_uploads)

        self._bg(fetch, done)

    # -------------------------------------------------------- local folders

    def _add_folder(self):
        d = filedialog.askdirectory(title="Folder containing rlogs")
        if d and d not in self.local_paths:
            self.local_paths.append(d)
            self.paths_list.insert("end", d)

    def _remove_folder(self):
        for i in reversed(self.paths_list.curselection()):
            self.local_paths.pop(i)
            self.paths_list.delete(i)

    # -------------------------------------------------------------- grading

    def _grade(self):
        connect = self.source_var.get() == "connect"
        routes = list(self.tree.selection()) if connect else []
        local_paths = [] if connect else list(self.local_paths)
        if not routes and not local_paths:
            messagebox.showinfo(
                "nothing selected",
                "Select one or more drives above (with rlogs ready)."
                if connect
                else "Add one or more local rlog folders above.",
            )
            return

        not_ready, partial = [], []
        for full in routes:
            b = self.badges.get(full)
            if not b or b["n_segments"] == 0 or b["n_logs"] == 0:
                not_ready.append(full)
            elif b["n_logs"] < b["n_segments"]:
                frac = b["n_logs"] / b["n_segments"]
                (partial if frac >= PARTIAL_OK_FRACTION else not_ready).append(full)
        if not_ready:
            messagebox.showwarning(
                "rlogs not uploaded",
                "These drives don't have enough rlogs uploaded yet "
                "(use Request upload first):\n\n"
                + "\n".join(f.split('|', 1)[-1] for f in not_ready),
            )
            return
        if partial and not messagebox.askyesno(
            "partial rlogs",
            "Some selected drives are missing a few segments (≥80% uploaded). "
            "The report will skip those minutes. Continue?",
        ):
            return

        targets = {}
        for p, var in self.tf_vars.items():
            try:
                v = float(var.get())
                if 0.3 <= v <= 5.0:
                    targets[p] = v
                else:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "bad follow target",
                    f"'{var.get()}' is not a valid t_follow for {p} (want 0.3-5.0 s)",
                )
                return
        set_t_follow(targets)  # persist for next time (and for the CLI)

        # Cheap pre-check from already-known route metadata (no download or
        # decode) -- a fast UX nicety, NOT a substitute for the pipeline's
        # authoritative post-decode gate below, which still runs regardless
        # and also catches model-only mismatches this can't see.
        allow_mixed = False
        if connect and len(routes) > 1:
            platforms = sorted({
                r["platform"] for r in self.routes
                if r["fullname"] in routes and r["platform"]
            })
            if len(platforms) > 1:
                if not messagebox.askyesno(
                    "different vehicles",
                    f"These routes are from different vehicles ({', '.join(platforms)}) "
                    "-- grade them together anyway?",
                ):
                    return
                allow_mixed = True

        self._launch_grade_job(routes, local_paths, targets, allow_mixed)

    def _launch_grade_job(self, routes: list[str], local_paths: list[str],
                          targets: dict, allow_mixed: bool):
        if not self.jobs.try_start(", ".join(routes + local_paths)):
            messagebox.showinfo("busy", "A grading job is already running — wait for it to finish.")
            return
        # remembered so the job-failure UI can offer a same-selection retry
        # with the override if the pipeline's post-decode gate rejects it
        self._last_grade_args = (routes, local_paths, targets)
        self.grade_btn.config(state="disabled")
        self.progress.config(mode="indeterminate")
        self.progress.start(80)
        threading.Thread(
            target=C.run_grade_job,
            args=(self.jobs, routes, local_paths, self.jwt, targets,
                  self.profile_var.get(), allow_mixed),
            daemon=True,
        ).start()
        self.root.after(POLL_JOB_MS, self._poll_job)

    def _poll_job(self):
        j = self.jobs.snapshot()
        detail = j.get("detail") or ""
        self.status.config(text=f"{j['phase']} — {detail}" if detail else j["phase"])
        prog = j.get("progress")
        if prog:
            self.progress.stop()
            self.progress.config(mode="determinate", maximum=max(1, prog[1]), value=prog[0])
        if j.get("active"):
            self.root.after(POLL_JOB_MS, self._poll_job)
            return
        self.progress.stop()
        self.progress.config(mode="determinate", value=0)
        self.grade_btn.config(state="normal")
        if j["phase"] == "done" and j.get("report"):
            self.status.config(text=f"report ready: {j['report']}")
            webbrowser.open(Path(j["report"]).as_uri())
            self._refresh_reports()
            self._refresh_profile()
        elif j["phase"] == "error" and j.get("error"):
            err = j["error"]
            self.status.config(text=f"error: {err['message']}")
            if err.get("type") == "MismatchError" and self._last_grade_args:
                routes, local_paths, targets = self._last_grade_args
                if messagebox.askyesno(
                    "grading failed",
                    err["message"] + "\n\nRetry with --allow-mixed (grade them together anyway)? "
                    "The report will carry a warning banner.",
                ):
                    self._launch_grade_job(routes, local_paths, targets, allow_mixed=True)
                return
            messagebox.showerror(
                "grading failed",
                err["message"] + "\n\n" + "\n".join(err.get("traceback") or []),
            )

    # -------------------------------------------------------------- reports

    def _refresh_reports(self):
        self.reports_list.delete(0, "end")
        self._report_paths = []
        for r in C.list_reports():
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["mtime"]))
            self.reports_list.insert("end", f"{r['name']}   ({when}, {r['size'] // 1024} KB)")
            self._report_paths.append(r["path"])
        self._bg(
            C.route_cache_size,
            lambda size, _err: size is not None
            and self.cache_label.config(text=f"rlog cache: {size / 1e9:.2f} GB"),
        )

    def _open_report(self, _event):
        sel = self.reports_list.curselection()
        if sel:
            webbrowser.open(Path(self._report_paths[sel[0]]).as_uri())

    def _delete_reports(self):
        sel = self.reports_list.curselection()
        if not sel:
            messagebox.showinfo("delete reports", "select one or more reports to delete first")
            return
        names = [Path(self._report_paths[i]).name for i in sel]
        if not messagebox.askyesno(
            "Delete reports",
            f"Delete {len(names)} report(s)?\n\n" + "\n".join(names[:8])
            + ("\n…" if len(names) > 8 else ""),
        ):
            return
        for i in sel:
            try:
                C.delete_report(self._report_paths[i])
            except C.ApiError as e:
                messagebox.showerror("delete failed", str(e))
        self._refresh_reports()

    def _clear_cache(self):
        def refresh_label(size, _err):
            if size is not None:
                self.cache_label.config(text=f"rlog cache: {size / 1e9:.2f} GB")

        def clear():
            return C.clear_route_cache()

        def done(freed, err):
            if err:
                messagebox.showerror("clear cache failed", str(err))
            else:
                messagebox.showinfo("cache cleared", f"freed {freed / 1e9:.2f} GB of downloaded rlogs")
            self._bg(C.route_cache_size, refresh_label)

        size = None
        try:
            size = C.route_cache_size()
        except OSError:
            pass
        if not messagebox.askyesno(
            "Clear downloaded rlogs",
            "Delete all downloaded rlogs from the cache"
            + (f" ({size / 1e9:.2f} GB)" if size else "")
            + "?\n\nReports are kept. Re-grading a drive will re-download its logs.",
        ):
            return
        self._bg(clear, done)

    # ------------------------------------------------------------- profile

    def _refresh_profile(self):
        def load():
            from . import profile as P

            return P.current_summary()

        def done(summary, err):
            if err or summary is None or summary.empty:
                self.profile_label.config(text="no profile yet")
            else:
                self.profile_label.config(text="; ".join(summary.lines()))
            self._fit_window()  # label text just changed length

        self._bg(load, done)

    def _delete_profile(self):
        from . import profile as P

        store = P.load_store()
        if not store.get("fingerprints"):
            messagebox.showinfo("driver profile", "there is no driver profile to delete.")
            return
        detail = "\n".join(P.describe_store(store))
        if not messagebox.askyesno(
            "Delete driver profile",
            "Permanently delete the local driver-baseline profile?\n\n"
            f"{detail}\n\nFuture grades will start with an empty baseline pool "
            "again (this run's own manual driving still grades normally).",
        ):
            return
        P.delete_profile()
        self._refresh_profile()


def _set_windows_dpi_awareness() -> None:
    """Without this, Tk on Windows reports widget sizes in unscaled units
    while the OS renders everything at the real (scaled) pixel size on any
    display above 100% scaling -- very common on Windows laptops. The window
    then gets fit too small for what's actually on screen, clipping the
    bottom row until the user manually resizes. Must run before Tk() exists.
    Best-effort: no-op (and harmless) on anything but Windows.
    """
    if not sys.platform.startswith("win"):
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # older Windows fallback
        except Exception:
            pass


def run() -> None:
    _set_windows_dpi_awareness()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
