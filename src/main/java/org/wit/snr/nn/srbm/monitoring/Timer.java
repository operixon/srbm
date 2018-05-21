package org.wit.snr.nn.srbm.monitoring;

import java.util.ArrayList;
import java.util.List;

public class Timer {

    private class Marker {
        final String description;
        final long timestamp;

        public Marker(String description, long timestamp) {
            this.description = description;
            this.timestamp = timestamp;
        }
    }

    public void start() {
        markers.add(new Marker("Start", System.currentTimeMillis()));
    }

    private List<Marker> markers = new ArrayList<>();

    public void mark(String description) {
        markers.add(new Marker(description, System.currentTimeMillis()));
    }

    public void reset() {
        markers.clear();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        double totalTime = (markers.get(markers.size() - 1).timestamp - markers.get(0).timestamp) / 1000.0;
        sb.append("| Total: ").append(totalTime).append("s ");
        for (int i = 1; i < markers.size(); i++) {
            long now = markers.get(i).timestamp;
            long before = markers.get(i - 1).timestamp;
            sb.append("| ").append(String.format("%s: %.2fs", markers.get(i).description, (now - before) / 1000.0)).append(" ");
        }
        return sb.toString();
    }
}
