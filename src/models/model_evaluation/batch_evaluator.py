"""
Batch Model Evaluator for MMA Betting Models

This module provides functionality to evaluate multiple models and compare their performance.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Import the main evaluator (assuming it's in the same directory)
from main_evaluator import MMABettingEvaluator, BettingConfig


class BatchModelEvaluator:
    """Evaluate multiple models and compare their performance"""

    def __init__(self, base_config: BettingConfig = None):
        """
        Initialize batch evaluator with base configuration

        Args:
            base_config: Base configuration to use for all models
        """
        self.base_config = base_config or BettingConfig()
        self.console = Console()
        self.results = []

    def evaluate_single_model(self, model_name: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single model

        Args:
            model_name: Name of the model file to evaluate
            verbose: Whether to print detailed output

        Returns:
            Dictionary of evaluation metrics
        """
        # Create a config for this specific model
        config = BettingConfig(
            val_data_path=self.base_config.val_data_path,
            test_data_path=self.base_config.test_data_path,
            encoder_path=self.base_config.encoder_path,
            model_base_path=self.base_config.model_base_path,
            output_dir=self.base_config.output_dir,
            initial_bankroll=self.base_config.initial_bankroll,
            confidence_threshold=self.base_config.confidence_threshold,
            kelly_fraction=self.base_config.kelly_fraction,
            fixed_bet_fraction=self.base_config.fixed_bet_fraction,
            max_bet_percentage=self.base_config.max_bet_percentage,
            min_odds=self.base_config.min_odds,
            max_underdog_odds=self.base_config.max_underdog_odds,
            use_ensemble=False,  # Single model evaluation
            use_calibration=self.base_config.use_calibration,
            calibration_type=self.base_config.calibration_type,
            odds_type=self.base_config.odds_type,
            model_files=[model_name]  # Just this model
        )

        # Create evaluator with quiet mode
        evaluator = MMABettingEvaluator(config)

        try:
            metrics = evaluator.run()

            # Add model name and remove unneeded nested dicts for CSV
            clean_metrics = {
                'model_name': model_name,
                'confidence_threshold': config.confidence_threshold,
                'calibration_type': config.calibration_type,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'auc': metrics.get('auc', 0) if metrics.get('auc') is not None else 0,
                'calibration_error': metrics.get('calibration_error', 0),
                'calibration_bias': metrics.get('calibration_bias', 0),
                'calibration_tendency': metrics.get('calibration_tendency', ''),
                'kelly_roi': metrics.get('kelly_roi', 0),
                'fixed_roi': metrics.get('fixed_roi', 0),
                'kelly_accuracy': metrics.get('kelly_accuracy', 0),
                'fixed_accuracy': metrics.get('fixed_accuracy', 0),
                'kelly_total_bets': metrics.get('kelly_total_bets', 0),
                'fixed_total_bets': metrics.get('fixed_total_bets', 0),
                'kelly_correct_bets': metrics.get('kelly_correct_bets', 0),
                'fixed_correct_bets': metrics.get('fixed_correct_bets', 0),
                'kelly_final_bankroll': metrics.get('kelly_final_bankroll', 0),
                'fixed_final_bankroll': metrics.get('fixed_final_bankroll', 0),
                'confident_predictions': metrics.get('confident_predictions', 0),
                'correct_confident_predictions': metrics.get('correct_confident_predictions', 0)
            }

            return clean_metrics

        except Exception as e:
            if verbose:
                self.console.print(f"[red]Error evaluating {model_name}: {str(e)}[/red]")
            return None

    def evaluate_directory(self, directory: str = None, output_filename: str = 'model_comparison.csv',
                          quiet: bool = True) -> pd.DataFrame:
        """
        Evaluate all models in a directory and save comprehensive metrics to CSV

        Args:
            directory: Directory containing model files (uses config default if None)
            output_filename: Name for the output CSV file
            quiet: If True, suppress progress output during evaluation

        Returns:
            DataFrame with comparison metrics for all models
        """
        directory = directory or self.base_config.model_base_path

        # Find all model files
        model_files = self._find_model_files(directory)

        if not model_files:
            self.console.print(f"[red]No model files found in {directory}[/red]")
            return pd.DataFrame()

        if not quiet:
            self.console.print(f"[cyan]Found {len(model_files)} models to evaluate[/cyan]")
            self.console.print(f"[cyan]Configuration: Calibration={self.base_config.calibration_type}, "
                              f"Kelly={self.base_config.kelly_fraction}, "
                              f"Threshold={self.base_config.confidence_threshold}[/cyan]\n")
        else:
            # Minimal output when quiet
            self.console.print(f"Evaluating {len(model_files)} models... (this may take a few minutes)")

        # Clear previous results
        self.results = []

        # Evaluate each model
        for i, model_file in enumerate(model_files, 1):
            if not quiet:
                self.console.print(f"[dim]Evaluating model {i}/{len(model_files)}: {model_file}[/dim]")

            metrics = self.evaluate_single_model(model_file, verbose=False)
            if metrics:
                self.results.append(metrics)

                if not quiet:
                    # Show quick progress update
                    if metrics['kelly_roi'] > 0:
                        self.console.print(f"  ✓ Kelly ROI: {metrics['kelly_roi']:.2f}%")
                    else:
                        self.console.print(f"  ✗ Kelly ROI: {metrics['kelly_roi']:.2f}%")

        # Create comprehensive DataFrame
        if self.results:
            df = pd.DataFrame(self.results)

            # Sort by Kelly ROI (descending)
            df = df.sort_values('kelly_roi', ascending=False)

            # Add ranking column
            df.insert(0, 'rank', range(1, len(df) + 1))

            # Reorder columns for better readability in CSV
            column_order = [
                'rank', 'model_name',
                'kelly_roi', 'fixed_roi',
                'kelly_final_bankroll', 'fixed_final_bankroll',
                'kelly_accuracy', 'fixed_accuracy',
                'kelly_total_bets', 'fixed_total_bets',
                'kelly_correct_bets', 'fixed_correct_bets',
                'accuracy', 'precision', 'recall', 'f1_score', 'auc',
                'calibration_error', 'calibration_bias', 'calibration_tendency',
                'confident_predictions', 'correct_confident_predictions',
                'confidence_threshold', 'calibration_type'
            ]

            # Ensure all columns exist (some might be missing)
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]

            # Save full results to CSV (always save, this is the main purpose)
            output_file = os.path.join(self.base_config.output_dir, output_filename)
            df.to_csv(output_file, index=False)

            # Also save a summary CSV with just key metrics
            summary_columns = [
                'rank', 'model_name', 'kelly_roi', 'fixed_roi',
                'kelly_accuracy', 'accuracy', 'auc', 'calibration_error'
            ]
            summary_df = df[[col for col in summary_columns if col in df.columns]]
            summary_file = output_file.replace('.csv', '_summary.csv')
            summary_df.to_csv(summary_file, index=False)

            # Final output message
            self.console.print(f"\n[bold green]✓ Evaluation complete![/bold green]")
            self.console.print(f"Results saved to: {output_file}")
            self.console.print(f"Summary saved to: {summary_file}")

            if not quiet:
                # Display summary statistics
                self.console.print("\n[bold cyan]Summary Statistics:[/bold cyan]")
                self.console.print(f"Total models evaluated: {len(df)}")
                self.console.print(f"Models with positive Kelly ROI: {len(df[df['kelly_roi'] > 0])}")
                self.console.print(f"Average Kelly ROI: {df['kelly_roi'].mean():.2f}%")
                self.console.print(f"Best Kelly ROI: {df['kelly_roi'].max():.2f}%")
                self.console.print(f"Worst Kelly ROI: {df['kelly_roi'].min():.2f}%")

                # Display top models
                self.console.print("\n[bold cyan]Top Models:[/bold cyan]")
                self._display_top_models(df)
            else:
                # Just show the best model when quiet
                best_model = df.iloc[0]
                self.console.print(f"Best model: {best_model['model_name']} (Kelly ROI: {best_model['kelly_roi']:.2f}%)")

            return df
        else:
            self.console.print("[red]No models were successfully evaluated[/red]")
            return pd.DataFrame()

    def _find_model_files(self, directory: str) -> List[str]:
        """Find all model files in directory"""
        model_files = []

        for file in os.listdir(directory):
            if file.endswith('.json'):  # XGBoost model files
                model_files.append(file)

        return sorted(model_files)

    def _display_top_models(self, df: pd.DataFrame, top_n: int = 10):
        """Display the top performing models"""
        table = Table(title=f"Top {min(top_n, len(df))} Models by Kelly ROI")

        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Model Name", style="white")
        table.add_column("Kelly ROI", justify="right", style="green")
        table.add_column("Fixed ROI", justify="right", style="yellow")
        table.add_column("Accuracy", justify="right", style="blue")
        table.add_column("AUC", justify="right", style="magenta")
        table.add_column("Cal. Error", justify="right", style="red")
        table.add_column("Tendency", style="white")

        for i, row in df.head(top_n).iterrows():
            rank = df.index.get_loc(i) + 1

            table.add_row(
                str(rank),
                row['model_name'][:40],  # Truncate long names
                f"{row['kelly_roi']:.2f}%",
                f"{row['fixed_roi']:.2f}%",
                f"{row['accuracy']:.4f}",
                f"{row.get('auc', 0):.4f}" if row.get('auc') is not None else "N/A",
                f"{row['calibration_error']:.4f}",
                row['calibration_tendency']
            )

        self.console.print(table)

    def compare_calibration_methods(self, model_name: str) -> pd.DataFrame:
        """
        Compare different calibration methods for a specific model

        Args:
            model_name: Name of the model to test

        Returns:
            DataFrame comparing calibration methods
        """
        calibration_types = ['uncalibrated', 'isotonic', 'range_based']
        comparison_results = []

        self.console.print(f"[cyan]Comparing calibration methods for {model_name}[/cyan]")

        for cal_type in calibration_types:
            self.console.print(f"Testing {cal_type} calibration...")

            # Update config for this calibration type
            config = BettingConfig(
                val_data_path=self.base_config.val_data_path,
                test_data_path=self.base_config.test_data_path,
                encoder_path=self.base_config.encoder_path,
                model_base_path=self.base_config.model_base_path,
                output_dir=self.base_config.output_dir,
                initial_bankroll=self.base_config.initial_bankroll,
                use_ensemble=False,
                use_calibration=(cal_type != 'uncalibrated'),
                calibration_type=cal_type if cal_type != 'uncalibrated' else None,
                model_files=[model_name]
            )

            evaluator = MMABettingEvaluator(config)

            try:
                metrics = evaluator.run()
                metrics['calibration_method'] = cal_type
                comparison_results.append({
                    'Calibration Method': cal_type,
                    'Kelly ROI': f"{metrics['kelly_roi']:.2f}%",
                    'Fixed ROI': f"{metrics['fixed_roi']:.2f}%",
                    'Calibration Error': f"{metrics['calibration_error']:.4f}",
                    'Tendency': metrics['calibration_tendency'],
                    'Kelly Accuracy': f"{metrics['kelly_accuracy']:.2%}",
                    'Total Bets': metrics['kelly_total_bets']
                })
            except Exception as e:
                self.console.print(f"[red]Error with {cal_type}: {str(e)}[/red]")

        # Display comparison
        if comparison_results:
            df = pd.DataFrame(comparison_results)

            table = Table(title=f"Calibration Method Comparison for {model_name}")

            for column in df.columns:
                table.add_column(column)

            for _, row in df.iterrows():
                table.add_row(*[str(val) for val in row.values])

            self.console.print(table)

            return df

        return pd.DataFrame()

    def parameter_sensitivity_analysis(self, model_name: str, quiet: bool = True) -> pd.DataFrame:
        """
        Analyze sensitivity to different parameter settings

        Args:
            model_name: Name of the model to analyze
            quiet: If True, suppress detailed output

        Returns:
            DataFrame with parameter sensitivity results
        """
        results = []

        # Test different Kelly fractions
        kelly_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

        if not quiet:
            self.console.print(f"[cyan]Testing Kelly fraction sensitivity for {model_name}[/cyan]")

        for kf in kelly_fractions:
            config = BettingConfig(
                val_data_path=self.base_config.val_data_path,
                test_data_path=self.base_config.test_data_path,
                encoder_path=self.base_config.encoder_path,
                model_base_path=self.base_config.model_base_path,
                kelly_fraction=kf,
                use_ensemble=False,
                model_files=[model_name]
            )

            evaluator = MMABettingEvaluator(config, quiet=quiet)

            try:
                metrics = evaluator.run()
                results.append({
                    'Kelly Fraction': kf,
                    'Kelly ROI': metrics['kelly_roi'],
                    'Kelly Final Bankroll': metrics['kelly_final_bankroll'],
                    'Kelly Bets': metrics['kelly_total_bets']
                })
            except Exception as e:
                if not quiet:
                    self.console.print(f"[red]Error with Kelly fraction {kf}: {str(e)}[/red]")

        if results:
            df = pd.DataFrame(results)

            if not quiet:
                # Display results
                table = Table(title="Kelly Fraction Sensitivity Analysis")
                table.add_column("Kelly Fraction", style="cyan")
                table.add_column("ROI", justify="right", style="green")
                table.add_column("Final Bankroll", justify="right", style="yellow")
                table.add_column("Total Bets", justify="right", style="blue")

                for _, row in df.iterrows():
                    table.add_row(
                        f"{row['Kelly Fraction']:.2f}",
                        f"{row['Kelly ROI']:.2f}%",
                        f"${row['Kelly Final Bankroll']:.2f}",
                        str(row['Kelly Bets'])
                    )

                self.console.print(table)

            return df

        return pd.DataFrame()


def main():
    """Main entry point for batch evaluation"""
    console = Console()

    # Create base configuration
    base_config = BettingConfig(
        use_calibration=True,
        calibration_type='isotonic',
        initial_bankroll=10000,
        kelly_fraction=0.5,
        fixed_bet_fraction=0.1,
        confidence_threshold=0.5
    )

    # Create batch evaluator
    batch_evaluator = BatchModelEvaluator(base_config)

    # Example 1: Evaluate all models in directory and save to CSV
    console.print("[bold cyan]Evaluating all models in directory...[/bold cyan]")
    comparison_df = batch_evaluator.evaluate_directory(
        output_filename='model_comparison_results.csv'  # Specify output filename
    )

    # Example 2: Compare calibration methods for the best model
    if not comparison_df.empty:
        best_model = comparison_df.iloc[0]['model_name']
        console.print(f"\n[bold cyan]Comparing calibration methods for best model: {best_model}[/bold cyan]")
        calibration_comparison = batch_evaluator.compare_calibration_methods(best_model)

        # Save calibration comparison to CSV
        if not calibration_comparison.empty:
            cal_output = os.path.join(base_config.output_dir, f'{best_model}_calibration_comparison.csv')
            calibration_comparison.to_csv(cal_output, index=False)
            console.print(f"[green]Calibration comparison saved to: {cal_output}[/green]")

    # Example 3: Parameter sensitivity analysis
    if not comparison_df.empty:
        console.print(f"\n[bold cyan]Running parameter sensitivity analysis...[/bold cyan]")
        sensitivity_df = batch_evaluator.parameter_sensitivity_analysis(best_model)

        # Save sensitivity analysis to CSV
        if not sensitivity_df.empty:
            sens_output = os.path.join(base_config.output_dir, f'{best_model}_sensitivity_analysis.csv')
            sensitivity_df.to_csv(sens_output, index=False)
            console.print(f"[green]Sensitivity analysis saved to: {sens_output}[/green]")

    console.print("\n[bold green]✓ All evaluations complete! Check the output directory for CSV files.[/bold green]")

    return comparison_df


if __name__ == "__main__":
    batch = BatchModelEvaluator()
    batch.evaluate_directory()
