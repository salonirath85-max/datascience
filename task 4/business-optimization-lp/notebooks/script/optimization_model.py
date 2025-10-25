"""
Production Planning Optimization Model
Solving a furniture manufacturing problem using Linear Programming
"""

import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ProductionOptimizer:
    """
    A class to solve production planning optimization problems using Linear Programming
    """
    
    def __init__(self):
        self.model = None
        self.variables = {}
        self.results = {}
        
    def setup_problem(self):
        """
        Define the optimization problem: maximize profit for furniture production
        """
        print("="*60)
        print("PRODUCTION PLANNING OPTIMIZATION")
        print("="*60)
        
        # Create the model
        self.model = pulp.LpProblem("Furniture_Production_Optimization", pulp.LpMaximize)
        
        # Decision variables (number of units to produce)
        self.variables['Chairs'] = pulp.LpVariable('Chairs', lowBound=50, cat='Integer')
        self.variables['Tables'] = pulp.LpVariable('Tables', lowBound=30, cat='Integer')
        self.variables['Desks'] = pulp.LpVariable('Desks', lowBound=20, cat='Integer')
        
        # Profit per unit
        profit = {'Chairs': 45, 'Tables': 80, 'Desks': 120}
        
        # Objective function: Maximize total profit
        self.model += (
            profit['Chairs'] * self.variables['Chairs'] +
            profit['Tables'] * self.variables['Tables'] +
            profit['Desks'] * self.variables['Desks'],
            "Total_Profit"
        )
        
        # Constraints
        
        # 1. Wood constraint (board feet)
        self.model += (
            5 * self.variables['Chairs'] +
            20 * self.variables['Tables'] +
            30 * self.variables['Desks'] <= 5000,
            "Wood_Constraint"
        )
        
        # 2. Metal constraint (kg)
        self.model += (
            2 * self.variables['Chairs'] +
            4 * self.variables['Tables'] +
            5 * self.variables['Desks'] <= 1500,
            "Metal_Constraint"
        )
        
        # 3. Fabric constraint (yards)
        self.model += (
            3 * self.variables['Chairs'] +
            2 * self.variables['Tables'] +
            0 * self.variables['Desks'] <= 800,
            "Fabric_Constraint"
        )
        
        # 4. Labor hours constraint
        self.model += (
            3 * self.variables['Chairs'] +
            5 * self.variables['Tables'] +
            8 * self.variables['Desks'] <= 2000,
            "Labor_Hours_Constraint"
        )
        
        # 5. Machine time constraint (hours)
        self.model += (
            2 * self.variables['Chairs'] +
            3 * self.variables['Tables'] +
            4 * self.variables['Desks'] <= 1200,
            "Machine_Time_Constraint"
        )
        
        # 6. Storage capacity constraint (total units)
        self.model += (
            self.variables['Chairs'] +
            self.variables['Tables'] +
            self.variables['Desks'] <= 500,
            "Storage_Capacity"
        )
        
        print("\n✓ Problem setup complete!")
        print(f"  - Decision Variables: {len(self.variables)}")
        print(f"  - Constraints: {len(self.model.constraints)}")
        
    def solve(self):
        """
        Solve the optimization problem
        """
        print("\n" + "="*60)
        print("SOLVING OPTIMIZATION PROBLEM")
        print("="*60)
        
        # Solve the problem
        self.model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Check solution status
        status = pulp.LpStatus[self.model.status]
        print(f"\nSolution Status: {status}")
        
        if status == 'Optimal':
            # Extract results
            self.results['Chairs'] = self.variables['Chairs'].varValue
            self.results['Tables'] = self.variables['Tables'].varValue
            self.results['Desks'] = self.variables['Desks'].varValue
            self.results['Total_Profit'] = pulp.value(self.model.objective)
            
            print("\n✓ Optimal solution found!")
            return True
        else:
            print("\n✗ No optimal solution found!")
            return False
    
    def display_results(self):
        """
        Display optimization results
        """
        print("\n" + "="*60)
        print("OPTIMAL PRODUCTION PLAN")
        print("="*60)
        
        print("\nProduction Quantities:")
        print(f"  Chairs:  {self.results['Chairs']:.0f} units")
        print(f"  Tables:  {self.results['Tables']:.0f} units")
        print(f"  Desks:   {self.results['Desks']:.0f} units")
        print(f"  Total:   {sum([self.results['Chairs'], self.results['Tables'], self.results['Desks']]):.0f} units")
        
        print(f"\nMaximum Profit: ${self.results['Total_Profit']:,.2f}")
        
        # Calculate resource utilization
        self.calculate_resource_utilization()
        
    def calculate_resource_utilization(self):
        """
        Calculate and display resource utilization
        """
        print("\n" + "="*60)
        print("RESOURCE UTILIZATION")
        print("="*60)
        
        chairs = self.results['Chairs']
        tables = self.results['Tables']
        desks = self.results['Desks']
        
        # Define resource usage
        resources = {
            'Wood (board ft)': {
                'used': 5*chairs + 20*tables + 30*desks,
                'capacity': 5000
            },
            'Metal (kg)': {
                'used': 2*chairs + 4*tables + 5*desks,
                'capacity': 1500
            },
            'Fabric (yards)': {
                'used': 3*chairs + 2*tables + 0*desks,
                'capacity': 800
            },
            'Labor (hours)': {
                'used': 3*chairs + 5*tables + 8*desks,
                'capacity': 2000
            },
            'Machine Time (hours)': {
                'used': 2*chairs + 3*tables + 4*desks,
                'capacity': 1200
            },
            'Storage (units)': {
                'used': chairs + tables + desks,
                'capacity': 500
            }
        }
        
        self.resource_data = []
        
        for resource, values in resources.items():
            utilization = (values['used'] / values['capacity']) * 100
            slack = values['capacity'] - values['used']
            
            self.resource_data.append({
                'Resource': resource,
                'Used': values['used'],
                'Capacity': values['capacity'],
                'Utilization (%)': utilization,
                'Slack': slack
            })
            
            print(f"\n{resource}:")
            print(f"  Used: {values['used']:.1f} / {values['capacity']:.0f}")
            print(f"  Utilization: {utilization:.1f}%")
            print(f"  Slack: {slack:.1f}")
    
    def sensitivity_analysis(self):
        """
        Perform sensitivity analysis on the solution
        """
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS")
        print("="*60)
        
        print("\nShadow Prices (Dual Values):")
        
        for name, constraint in self.model.constraints.items():
            shadow_price = constraint.pi
            if shadow_price is not None and abs(shadow_price) > 0.001:
                print(f"  {name}: ${shadow_price:.2f}")
                print(f"    → Increasing this resource by 1 unit increases profit by ${shadow_price:.2f}")
        
        # Analyze profit changes with parameter variations
        self.profit_sensitivity()
    
    def profit_sensitivity(self):
        """
        Analyze how profit changes with price variations
        """
        print("\n" + "-"*60)
        print("Profit Sensitivity to Price Changes:")
        print("-"*60)
        
        base_prices = {'Chairs': 45, 'Tables': 80, 'Desks': 120}
        
        for product in ['Chairs', 'Tables', 'Desks']:
            print(f"\n{product}:")
            
            # Test price variations
            for change in [-20, -10, 0, 10, 20]:
                new_price = base_prices[product] + change
                profit_change = change * self.results[product]
                new_profit = self.results['Total_Profit'] + profit_change
                
                print(f"  Price ${new_price}: Total Profit = ${new_profit:,.2f} ({profit_change:+,.0f})")
    
    def create_visualizations(self, save_dir='visualizations'):
        """
        Create and save visualizations
        """
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Create directory if it doesn't exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Visualization 1: Production Quantities
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Production quantities bar chart
        products = ['Chairs', 'Tables', 'Desks']
        quantities = [self.results[p] for p in products]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        axes[0, 0].bar(products, quantities, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Optimal Production Quantities', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Units')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(quantities):
            axes[0, 0].text(i, v + 5, f'{v:.0f}', ha='center', fontweight='bold')
        
        # Resource utilization
        resource_df = pd.DataFrame(self.resource_data)
        
        axes[0, 1].barh(resource_df['Resource'], resource_df['Utilization (%)'], color='#9b59b6', alpha=0.7)
        axes[0, 1].set_xlabel('Utilization (%)')
        axes[0, 1].set_title('Resource Utilization', fontsize=14, fontweight='bold')
        axes[0, 1].axvline(x=100, color='red', linestyle='--', label='Capacity')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Profit breakdown
        profit_per_product = {
            'Chairs': 45 * self.results['Chairs'],
            'Tables': 80 * self.results['Tables'],
            'Desks': 120 * self.results['Desks']
        }
        
        axes[1, 0].pie(profit_per_product.values(), labels=profit_per_product.keys(), 
                       autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 0].set_title('Profit Contribution by Product', fontsize=14, fontweight='bold')
        
        # Resource slack
        axes[1, 1].bar(resource_df['Resource'], resource_df['Slack'], color='#1abc9c', alpha=0.7)
        axes[1, 1].set_ylabel('Slack (Unused Capacity)')
        axes[1, 1].set_title('Resource Slack', fontsize=14, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filepath = Path(save_dir) / 'optimization_results.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {filepath}")
        plt.close()
        
        # Visualization 2: Sensitivity Analysis
        self.create_sensitivity_chart(save_dir)
    
    def create_sensitivity_chart(self, save_dir):
        """
        Create sensitivity analysis visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Price sensitivity
        products = ['Chairs', 'Tables', 'Desks']
        base_prices = {'Chairs': 45, 'Tables': 80, 'Desks': 120}
        
        for idx, product in enumerate(products):
            price_changes = range(-20, 25, 5)
            profits = []
            
            for change in price_changes:
                profit_change = change * self.results[product]
                profits.append(self.results['Total_Profit'] + profit_change)
            
            axes[0].plot([base_prices[product] + c for c in price_changes], profits, 
                        marker='o', label=product, linewidth=2)
        
        axes[0].set_xlabel('Price ($)', fontsize=12)
        axes[0].set_ylabel('Total Profit ($)', fontsize=12)
        axes[0].set_title('Profit Sensitivity to Price Changes', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Resource impact
        resource_names = [rd['Resource'].split()[0] for rd in self.resource_data]
        utilization = [rd['Utilization (%)'] for rd in self.resource_data]
        
        colors_map = ['red' if u > 90 else 'orange' if u > 70 else 'green' for u in utilization]
        
        axes[1].barh(resource_names, utilization, color=colors_map, alpha=0.7)
        axes[1].axvline(x=90, color='orange', linestyle='--', label='High Utilization (90%)')
        axes[1].axvline(x=100, color='red', linestyle='--', label='Full Capacity')
        axes[1].set_xlabel('Utilization (%)', fontsize=12)
        axes[1].set_title('Resource Constraint Analysis', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        filepath = Path(save_dir) / 'sensitivity_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def save_results(self, output_dir='data/output'):
        """
        Save results to CSV files
        """
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save optimal solution
        solution_df = pd.DataFrame({
            'Product': ['Chairs', 'Tables', 'Desks', 'Total Profit'],
            'Quantity': [
                self.results['Chairs'],
                self.results['Tables'],
                self.results['Desks'],
                0
            ],
            'Value': [
                45 * self.results['Chairs'],
                80 * self.results['Tables'],
                120 * self.results['Desks'],
                self.results['Total_Profit']
            ]
        })
        
        filepath1 = Path(output_dir) / 'optimal_solution.csv'
        solution_df.to_csv(filepath1, index=False)
        print(f"\n✓ Saved: {filepath1}")
        
        # Save resource utilization
        resource_df = pd.DataFrame(self.resource_data)
        filepath2 = Path(output_dir) / 'resource_utilization.csv'
        resource_df.to_csv(filepath2, index=False)
        print(f"✓ Saved: {filepath2}")
    
    def generate_report(self):
        """
        Generate a comprehensive text report
        """
        print("\n" + "="*60)
        print("OPTIMIZATION REPORT SUMMARY")
        print("="*60)
        
        print("\n📊 EXECUTIVE SUMMARY")
        print("-" * 60)
        print(f"Optimal total profit: ${self.results['Total_Profit']:,.2f}")
        print(f"Total units produced: {sum([self.results['Chairs'], self.results['Tables'], self.results['Desks']]):.0f}")
        
        print("\n🏭 PRODUCTION RECOMMENDATIONS")
        print("-" * 60)
        print(f"• Produce {self.results['Chairs']:.0f} Chairs")
        print(f"• Produce {self.results['Tables']:.0f} Tables")
        print(f"• Produce {self.results['Desks']:.0f} Desks")
        
        print("\n⚠️  CRITICAL INSIGHTS")
        print("-" * 60)
        
        # Identify bottleneck resources
        resource_df = pd.DataFrame(self.resource_data)
        high_util = resource_df[resource_df['Utilization (%)'] > 90]
        
        if not high_util.empty:
            print("High utilization resources (>90%):")
            for _, row in high_util.iterrows():
                print(f"  • {row['Resource']}: {row['Utilization (%)']:.1f}% utilized")
        
        # Identify underutilized resources
        low_util = resource_df[resource_df['Utilization (%)'] < 50]
        if not low_util.empty:
            print("\nUnderutilized resources (<50%):")
            for _, row in low_util.iterrows():
                print(f"  • {row['Resource']}: {row['Utilization (%)']:.1f}% utilized")
        
        print("\n💡 STRATEGIC RECOMMENDATIONS")
        print("-" * 60)
        print("1. Focus on increasing wood supply (primary constraint)")
        print("2. Consider premium pricing for desks (highest profit margin)")
        print("3. Optimize labor allocation to high-value products")
        print("4. Explore opportunities to utilize excess fabric capacity")


def main():
    """
    Main execution function
    """
    # Initialize optimizer
    optimizer = ProductionOptimizer()
    
    # Setup and solve the problem
    optimizer.setup_problem()
    
    if optimizer.solve():
        # Display results
        optimizer.display_results()
        
        # Perform sensitivity analysis
        optimizer.sensitivity_analysis()
        
        # Create visualizations
        optimizer.create_visualizations()
        
        # Save results
        optimizer.save_results()
        
        # Generate comprehensive report
        optimizer.generate_report()
        
        print("\n" + "="*60)
        print("✅ OPTIMIZATION COMPLETE!")
        print("="*60)
        print("\nAll results have been saved to:")
        print("  • data/output/")
        print("  • visualizations/")
    

if __name__ == "__main__":
    main()