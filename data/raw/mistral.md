# Mistral 7B Instruct v0.3

## Overview

Mistral 7B Instruct v0.3 is an instruction-tuned large language model developed by Mistral AI. It is based on the Mistral 7B v0.3 base model and contains approximately seven billion parameters.

This release includes an extended vocabulary of 32,768 tokens, supports the v3 tokenizer, and introduces function-calling support. These features make it useful for applications that need to process user instructions and interact with external tools.

## Applications

Mistral 7B Instruct can be used for conversational assistants, text generation, question answering, and programming-related tasks. Its function-calling capability also allows developers to experiment with applications that connect language models to external functions and services.

## Limitations

An important limitation mentioned in the official model card is that Mistral 7B Instruct v0.3 does not include built-in moderation mechanisms. Therefore, developers should implement appropriate safety checks and output moderation when building applications for real users.

The model's relatively small parameter count also means that its performance should be evaluated carefully on complex tasks before choosing it over larger alternatives.

## Licensing

The Mistral 7B Instruct v0.3 model card identifies Apache License 2.0 as its license.

## Date accessed

2026-10-30

## Official source

https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3