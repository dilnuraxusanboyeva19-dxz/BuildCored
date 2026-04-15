import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# External Libraries
import git
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

console = Console()

class DailyDebrief:
    def __init__(self, repo_path=".", model="llama3"):
        self.repo_path = Path(repo_path).absolute()
        self.model = model
        
    def get_git_activity(self):
        """Collects commit messages from the last 24 hours."""
        try:
            repo = git.Repo(self.repo_path)
            since = datetime.now() - timedelta(days=1)
            commits = list(repo.iter_commits(since=since))
            return [f"Commit: {c.summary}" for c in commits]
        except Exception as e:
            return [f"Git Error: {str(e)}"]

    def get_file_changes(self):
        """Finds files modified in the last 24 hours."""
        recent_files = []
        one_day_ago = time.time() - (24 * 3600)
        
        for file in self.repo_path.rglob('*'):
            if file.is_file() and ".git" not in str(file):
                if file.stat().st_mtime > one_day_ago:
                    recent_files.append(f"Modified: {file.name}")
        
        return recent_files[:10]  

    def query_llm(self, context):
        """Sends data to local Ollama instance."""
        prompt = f"""
        You are a developer productivity assistant. Based on the following raw activity data from the last 24 hours:
        {context}

        Provide a strictly 5-line structured debrief:
        1. WHAT I BUILT: (1 sentence)
        2. WHAT BROKE/CHALLENGES: (1 sentence)
        3. WHAT I LEARNED: (1 sentence)
        4. WHAT'S NEXT: (1 sentence)
        5. OVERALL STATUS: (Success/Progressing/Blocked)
        
        Do not include any other text.
        """
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json().get("response", "No response from LLM.")
        except requests.exceptions.ConnectionError:
            return "Error: Ollama is not running. Please start Ollama."

    def run(self):
        console.print("[bold blue]Gathering daily telemetry...[/bold blue]")
        
        with Progress() as progress:
            t1 = progress.add_task("[green]Analyzing Git...", total=1)
            git_data = self.get_git_activity()
            progress.update(t1, advance=1)
            
            t2 = progress.add_task("[green]Scanning Files...", total=1)
            file_data = self.get_file_changes()
            progress.update(t2, advance=1)

        combined_data = "\n".join(git_data + file_data)
        
        if not combined_data:
            console.print("[yellow]No recent activity detected.[/yellow]")
            return

        console.print("[bold cyan]Generating AI Summary...[/bold cyan]")
        summary = self.query_llm(combined_data)
        
        console.print("\n")
        console.print(Panel(summary, title="[bold green]Daily Debrief[/bold green]", expand=False))

if __name__ == "__main__":
    # Update this line to match your 'qwen2.5:3b' model
    debrief = DailyDebrief(model="qwen2.5:3b") 
    debrief.run()
