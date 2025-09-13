#!/bin/bash
# Quick launch script for different growth charts

echo "🧲 Magnetic Curve Tracer - Chart Selector"
echo "===========================================" 

# List available image files
echo "Available chart images:"
find . -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" | sort

echo ""
echo "Usage examples:"
echo "  python magnetic_curve_tracer.py                                    # Default (boys)"
echo "  python magnetic_curve_tracer.py \"Thai Growth Chart Girls 2-19 Years.png\""
echo "  python magnetic_curve_tracer.py \"your_chart_name.png\""
echo ""
echo "Files will be saved as:"
echo "  [chart_name]_curves.json"
echo ""

# If argument provided, run with that image
if [ $# -eq 1 ]; then
    echo "🚀 Starting tracer for: $1"
    python magnetic_curve_tracer.py "$1"
else
    echo "💡 Tip: Pass an image filename as argument to start tracing!"
fi