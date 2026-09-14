# Stage 1: Build application with Maven and cached dependencies
FROM maven:3.9.6-eclipse-temurin-17 AS build
WORKDIR /app

# Cache Maven dependencies
COPY pom.xml .
RUN mvn dependency:go-offline -B || true

# Copy source code and build production jar
COPY src ./src
RUN mvn clean package -DskipTests -B

# Stage 2: Minimal, secure runtime
FROM eclipse-temurin:17-jre
WORKDIR /app

# Non-root system user for container security
RUN groupadd -r nagarseva && useradd -r -g nagarseva nagarseva

COPY --from=build /app/target/*.jar app.jar
RUN chown -R nagarseva:nagarseva /app

USER nagarseva

EXPOSE 8080
ENV PORT=8080
ENV SPRING_PROFILES_ACTIVE=prod

ENTRYPOINT ["java", "-Djava.security.egd=file:/dev/./urandom", "-jar", "app.jar"]