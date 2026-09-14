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
        normalizeDatabaseUrl();
        SpringApplication.run(NagarSevaApplication.class, args);
    }

    private static void normalizeDatabaseUrl() {
        String rawUrl = System.getenv("DATABASE_URL");
        if (rawUrl == null || rawUrl.isBlank()) {
            rawUrl = System.getProperty("DATABASE_URL");
        }
        if (rawUrl == null || rawUrl.isBlank() || rawUrl.startsWith("jdbc:h2:")) {
            return;
        }

        try {
            if (rawUrl.startsWith("postgres://") || rawUrl.startsWith("postgresql://")) {
                String normalizedUriStr = rawUrl.startsWith("postgres://")
                        ? "postgresql://" + rawUrl.substring("postgres://".length())
                        : rawUrl;

                java.net.URI uri = new java.net.URI(normalizedUriStr);
                String host = uri.getHost();
                int port = uri.getPort() > 0 ? uri.getPort() : 5432;
                String path = uri.getPath();
                String userInfo = uri.getUserInfo();
                String query = uri.getQuery();

                StringBuilder jdbcUrl = new StringBuilder("jdbc:postgresql://")
                        .append(host)
                        .append(":")
                        .append(port)
                        .append(path);

                if (query != null && !query.isBlank()) {
                    jdbcUrl.append("?").append(query);
                } else if (host != null && (host.contains("render.com") || host.contains("neon.tech") || host.contains("supabase.co") || host.contains("railway.app"))) {
                    jdbcUrl.append("?sslmode=require");
                }

                String finalJdbc = jdbcUrl.toString();
                System.setProperty("spring.datasource.url", finalJdbc);
                System.setProperty("DATABASE_URL", finalJdbc);
                System.setProperty("spring.datasource.driver-class-name", "org.postgresql.Driver");

                if (userInfo != null && userInfo.contains(":")) {
                    String[] parts = userInfo.split(":", 2);
                    System.setProperty("spring.datasource.username", parts[0]);
                    System.setProperty("spring.datasource.password", parts[1]);
                }
                log.info("Converted cloud DATABASE_URL to JDBC format for host '{}'", host);
            } else if (rawUrl.startsWith("jdbc:postgres://")) {
                String fixed = "jdbc:postgresql://" + rawUrl.substring("jdbc:postgres://".length());
                System.setProperty("spring.datasource.url", fixed);
                System.setProperty("DATABASE_URL", fixed);
                System.setProperty("spring.datasource.driver-class-name", "org.postgresql.Driver");
            } else if (!rawUrl.startsWith("jdbc:")) {
                String fixed = "jdbc:" + rawUrl;
                System.setProperty("spring.datasource.url", fixed);
                System.setProperty("DATABASE_URL", fixed);
            }
        } catch (Exception e) {
            log.warn("Could not normalize cloud DATABASE_URL: {}", e.getMessage());
        }
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
