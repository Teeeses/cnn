package ru.yrgu.web;

public class ProbabilityDTO {

    private String label;
    private Float value;

    public ProbabilityDTO() {
    }

    public ProbabilityDTO(String label, Float value) {
        this.label = label;
        this.value = value;
    }

    public String getLabel() {
        return label;
    }

    public ProbabilityDTO setLabel(String label) {
        this.label = label;
        return this;
    }

    public Float getValue() {
        return value;
    }

    public ProbabilityDTO setValue(Float value) {
        this.value = value;
        return this;
    }
}
