# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

The SochDB team takes security bugs seriously. We appreciate your efforts to
responsibly disclose your findings, and will make every effort to acknowledge
your contributions.

### How to Report a Security Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to **sushanth@sochdb.dev**.

You should receive a response within **48 hours**. If for some reason you do not,
please follow up via email to ensure we received your original message.

### What to Include in Your Report

Please include the following information in your report:

* Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

## Response Timeline

* **Initial Response**: Within 48 hours of report submission
* **Triage & Assessment**: Within 5 business days
* **Status Updates**: Every 7 days until resolution
* **Fix Development**: Depends on severity and complexity
* **Public Disclosure**: After fix is released and users have had time to update

## Severity Classification

We use the following severity levels:

* **Critical**: Exploitable vulnerability that could lead to remote code execution,
  data loss, or complete system compromise
* **High**: Significant security issue that affects data integrity or availability
* **Medium**: Security issue with limited scope or requiring specific conditions
* **Low**: Minor security concern with minimal impact

## Response Actions

Based on severity:

* **Critical**: Immediate response, hotfix within 48-72 hours if possible
* **High**: Response within 1 week, fix in next patch release
* **Medium**: Response within 2 weeks, fix in next minor release
* **Low**: Response within 30 days, fix in next planned release

## Disclosure Policy

* Security issues are kept confidential until a fix is released
* We will coordinate disclosure timing with the reporter
* Credit will be given to reporters unless anonymity is requested
* A security advisory will be published on GitHub after the fix is released

## Security Update Process

When a security fix is released:

1. A security advisory is published on GitHub
2. Release notes clearly indicate security fixes
3. Users are notified through our communication channels
4. CVE identifiers are assigned when applicable

## Bug Bounty Program

At this time, SochDB does not have a formal bug bounty program. However, we
deeply appreciate security researchers who help make SochDB more secure and
will publicly acknowledge contributors who wish to be recognized.

## Security Best Practices

When using SochDB:

* Always use the latest stable version
* Keep dependencies up to date
* Follow the security guidelines in our documentation
* Use TLS/SSL for network communications
* Implement proper authentication and authorization
* Regularly backup your data
* Monitor your deployments for suspicious activity

## Contact

For security-related questions or concerns:

* **Email**: sushanth@sochdb.dev
* **PGP Key**: Available upon request

Thank you for helping keep SochDB and our users safe!
