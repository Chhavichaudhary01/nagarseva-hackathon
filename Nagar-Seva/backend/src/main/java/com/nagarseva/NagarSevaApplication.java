package com.nagarseva;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

@SpringBootApplication
@EnableScheduling
public class NagarSevaApplication {

    private static final Logger log = LoggerFactory.getLogger(NagarSevaApplication.class);

    public static void main(String[] args) {
        loadDotEnvIfPresent();
        SpringApplication.run(NagarSevaApplication.class, args);
    }

    private static void loadDotEnvIfPresent() {
        Path[] candidatePaths = new Path[]{
                Path.of(".env"),
                Path.of(".env.local"),
                Path.of("backend", ".env"),
                Path.of("..", ".env"),
                Path.of("..", ".env.local"),
                Path.of("..", "..", ".env")
        };

        for (Path path : candidatePaths) {
            File f = path.toFile();
            if (f.exists() && f.isFile()) {
                try {
                    List<String> lines = Files.readAllLines(f.toPath());
                    int loadedCount = 0;
                    for (String line : lines) {
                        line = line.trim();
                        if (line.isEmpty() || line.startsWith("#")) {
                            continue;
                        }
                        int eqIdx = line.indexOf('=');
                        if (eqIdx > 0) {
                            String key = line.substring(0, eqIdx).trim();
                            String val = line.substring(eqIdx + 1).trim();
                            if ((val.startsWith("\"") && val.endsWith("\"")) ||
                                    (val.startsWith("'") && val.endsWith("'"))) {
                                val = val.substring(1, val.length() - 1);
                            }
                            if (!key.isEmpty() && System.getenv(key) == null && System.getProperty(key) == null) {
                                System.setProperty(key, val);
                                loadedCount++;
                            }
                        }
                    }
                    if (loadedCount > 0) {
                        log.info("Loaded {} environment properties from {}", loadedCount, f.getAbsolutePath());
                    }
                    break;
                } catch (IOException e) {
                    log.warn("Failed to read environment file at {}: {}", f.getAbsolutePath(), e.getMessage());
                }
            }
        }
    }

}
