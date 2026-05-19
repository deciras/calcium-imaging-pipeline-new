cd /mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_Suite2p_Results

find . -type f \( \
    -name "*_brightness_trace.csv" -o \
    -name "stim_map.csv" -o \
    -name "stim_events.csv" -o \
    -name "*_stim_trace.png" \
\) -exec trash {} +
