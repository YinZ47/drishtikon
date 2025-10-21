package com.drishtikon.stocksync.util;

import java.util.Optional;
import java.util.function.Function;

/**
 * Generic Result wrapper for handling success/failure states.
 * Day 1-2: Generic wrapper to avoid throwing exceptions for expected failures.
 */
public final class Result<T> {
    private final T value;
    private final String error;
    private final boolean success;

    private Result(T value, String error, boolean success) {
        this.value = value;
        this.error = error;
        this.success = success;
    }

    public static <T> Result<T> success(T value) {
        return new Result<>(value, null, true);
    }

    public static <T> Result<T> failure(String error) {
        return new Result<>(null, error, false);
    }

    public boolean isSuccess() {
        return success;
    }

    public boolean isFailure() {
        return !success;
    }

    public Optional<T> getValue() {
        return Optional.ofNullable(value);
    }

    public T getOrThrow() {
        if (!success) {
            throw new IllegalStateException("Result is a failure: " + error);
        }
        return value;
    }

    public T getOrDefault(T defaultValue) {
        return success ? value : defaultValue;
    }

    public String getError() {
        return error;
    }

    public <U> Result<U> map(Function<T, U> mapper) {
        if (success) {
            return Result.success(mapper.apply(value));
        }
        return Result.failure(error);
    }

    public <U> Result<U> flatMap(Function<T, Result<U>> mapper) {
        if (success) {
            return mapper.apply(value);
        }
        return Result.failure(error);
    }

    @Override
    public String toString() {
        if (success) {
            return "Result.Success(" + value + ")";
        } else {
            return "Result.Failure(" + error + ")";
        }
    }
}
