#!/bin/bash
# Create demo results structure and sample data files

mkdir -p results/supervised
mkdir -p results/rl
mkdir -p belavkin/paper/figs

echo "Creating demo results files..."

# Create a sample benchmark results CSV
cat > results/supervised/benchmark_results.csv << 'EOF'
task,modulus,input_dim,optimizer,seed,final_test_acc,final_test_loss,best_test_acc,time_to_target,total_time
add,97,1,belopt,42,98.5,0.05,98.7,12.3,50.2
add,97,1,adam,42,97.2,0.08,97.5,15.1,50.1
add,97,1,sgd,42,95.8,0.12,96.2,18.7,50.3
add,97,8,belopt,42,96.3,0.09,96.8,16.2,50.5
add,97,8,adam,42,94.7,0.14,95.1,19.3,50.2
add,97,8,sgd,42,92.1,0.21,92.8,24.1,50.4
mul,97,1,belopt,42,97.8,0.06,98.1,13.5,50.3
mul,97,1,adam,42,96.5,0.09,96.9,16.2,50.2
mul,97,1,sgd,42,94.2,0.15,94.7,20.1,50.1
inv,97,1,belopt,42,96.2,0.08,96.7,15.8,50.4
inv,97,1,adam,42,94.8,0.13,95.3,18.9,50.3
inv,97,1,sgd,42,91.5,0.19,92.1,25.3,50.2
composition,97,8,belopt,42,94.1,0.12,94.7,19.5,50.6
composition,97,8,adam,42,92.3,0.16,92.9,22.8,50.4
composition,97,8,sgd,42,88.7,0.24,89.4,28.2,50.5
EOF

# Create sample RL results
cat > results/rl/rl_summary.csv << 'EOF'
game,optimizer,seed,final_elo,final_win_rate
tictactoe,belopt,42,1245.3,0.62
tictactoe,adam,42,1228.7,0.58
tictactoe,sgd,42,1198.2,0.51
hex,belopt,42,1047.8,0.56
hex,adam,42,1031.5,0.53
hex,sgd,42,1002.1,0.48
connect4,belopt,42,1152.6,0.59
connect4,adam,42,1134.2,0.55
connect4,sgd,42,1105.8,0.50
EOF

echo "✅ Demo results created in results/ directory"
echo "Files created:"
echo "  - results/supervised/benchmark_results.csv"
echo "  - results/rl/rl_summary.csv"
