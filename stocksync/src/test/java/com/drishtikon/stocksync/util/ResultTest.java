package com.drishtikon.stocksync.util;

import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Result wrapper - Day 1-2
 */
class ResultTest {

    @Test
    void testSuccessResult() {
        Result<String> result = Result.success("test value");
        
        assertTrue(result.isSuccess());
        assertFalse(result.isFailure());
        assertEquals("test value", result.getOrThrow());
        assertEquals(Optional.of("test value"), result.getValue());
    }

    @Test
    void testFailureResult() {
        Result<String> result = Result.failure("error message");
        
        assertFalse(result.isSuccess());
        assertTrue(result.isFailure());
        assertEquals("error message", result.getError());
        assertEquals(Optional.empty(), result.getValue());
    }

    @Test
    void testGetOrThrowOnFailure() {
        Result<String> result = Result.failure("error");
        
        assertThrows(IllegalStateException.class, result::getOrThrow);
    }

    @Test
    void testGetOrDefault() {
        Result<String> success = Result.success("value");
        Result<String> failure = Result.failure("error");
        
        assertEquals("value", success.getOrDefault("default"));
        assertEquals("default", failure.getOrDefault("default"));
    }

    @Test
    void testMap() {
        Result<Integer> result = Result.success(5);
        Result<String> mapped = result.map(i -> "Number: " + i);
        
        assertTrue(mapped.isSuccess());
        assertEquals("Number: 5", mapped.getOrThrow());
    }

    @Test
    void testMapOnFailure() {
        Result<Integer> result = Result.failure("error");
        Result<String> mapped = result.map(i -> "Number: " + i);
        
        assertTrue(mapped.isFailure());
        assertEquals("error", mapped.getError());
    }

    @Test
    void testFlatMap() {
        Result<Integer> result = Result.success(5);
        Result<Integer> mapped = result.flatMap(i -> Result.success(i * 2));
        
        assertTrue(mapped.isSuccess());
        assertEquals(10, mapped.getOrThrow());
    }

    @Test
    void testFlatMapWithFailure() {
        Result<Integer> result = Result.success(5);
        Result<Integer> mapped = result.flatMap(i -> Result.failure("multiplied failed"));
        
        assertTrue(mapped.isFailure());
        assertEquals("multiplied failed", mapped.getError());
    }
}
