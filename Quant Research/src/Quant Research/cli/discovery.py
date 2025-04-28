# src/quant_research/providers/cli/discovery.py
import argparse
import json
import sys
from typing import Dict, Any, List

from ..provider_discovery import ProviderDiscovery, ProviderCapability
from ...core.config import ProviderType


def create_discovery_parser(subparsers):
    """Create CLI parser for provider discovery commands"""
    discovery_parser = subparsers.add_parser(
        'discover', 
        help='Discover available providers and their capabilities'
    )
    
    # Create subcommands for discovery
    discover_subparsers = discovery_parser.add_subparsers(
        dest='discover_command',
        help='Discovery subcommands'
    )
    
    # List providers command
    list_parser = discover_subparsers.add_parser(
        'list',
        help='List all available providers'
    )
    list_parser.add_argument(
        '--format', 
        choices=['table', 'json'], 
        default='table',
        help='Output format'
    )
    
    # Provider info command
    info_parser = discover_subparsers.add_parser(
        'info',
        help='Get detailed information about a provider'
    )
    info_parser.add_argument(
        'provider_id',
        help='Provider identifier'
    )
    info_parser.add_argument(
        '--format', 
        choices=['table', 'json'], 
        default='table',
        help='Output format'
    )
    
    # Find providers by type command
    by_type_parser = discover_subparsers.add_parser(
        'by-type',
        help='Find providers by type'
    )
    by_type_parser.add_argument(
        'provider_type',
        help='Provider type to search for'
    )
    by_type_parser.add_argument(
        '--format', 
        choices=['table', 'json'], 
        default='table',
        help='Output format'
    )
    
    # Find providers by capability command
    by_capability_parser = discover_subparsers.add_parser(
        'by-capability',
        help='Find providers by capability'
    )
    by_capability_parser.add_argument(
        'capability',
        help='Capability to search for'
    )
    by_capability_parser.add_argument(
        '--format', 
        choices=['table', 'json'], 
        default='table',
        help='Output format'
    )
    
    # Example command
    example_parser = discover_subparsers.add_parser(
        'example',
        help='Generate example code for a provider'
    )
    example_parser.add_argument(
        'provider_id',
        help='Provider identifier'
    )
    
    # Capabilities command
    capabilities_parser = discover_subparsers.add_parser(
        'capabilities',
        help='List all available capability types'
    )
    capabilities_parser.add_argument(
        '--format', 
        choices=['table', 'json'], 
        default='table',
        help='Output format'
    )
    
    return discovery_parser


def handle_discovery_command(args):
    """Handle provider discovery commands"""
    if args.discover_command == 'list':
        providers = ProviderDiscovery.get_available_providers()
        if args.format == 'json':
            print(json.dumps(providers, indent=2, default=str))
        else:
            # Table format
            print("Available Providers:")
            print("--------------------")
            for provider_id, info in providers.items():
                print(f"ID: {provider_id}")
                print(f"Type: {info['provider_type']}")
                print(f"Description: {info['description']}")
                print(f"Capabilities: {', '.join(str(c) for c in info['capabilities'])}")
                print()
    
    elif args.discover_command == 'info':
        try:
            info = ProviderDiscovery.get_provider_info(args.provider_id)
            if args.format == 'json':
                print(json.dumps(info, indent=2, default=str))
            else:
                # Table format
                print(f"Provider: {args.provider_id}")
                print("-" * (len(args.provider_id) + 10))
                print(f"Type: {info['provider_type']}")
                print(f"Description: {info['description']}")
                print(f"Class: {info['class_name']}")
                print(f"Module: {info['module']}")
                print(f"Capabilities: {', '.join(str(c) for c in info['capabilities'])}")
                print("\nMethods:")
                for method in info['methods']:
                    print(f"- {method}")
                
                print("\nConfiguration Schema:")
                for field, field_info in info['config_schema'].get('properties', {}).items():
                    required = " (required)" if field in info['config_schema'].get('required_fields', []) else ""
                    print(f"- {field}{required}: {field_info['description']}")
                
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    elif args.discover_command == 'by-type':
        try:
            # Convert string to ProviderType enum
            provider_type = args.provider_type
            for pt in ProviderType:
                if pt.value == provider_type:
                    provider_type = pt
                    break
            
            providers = ProviderDiscovery.find_providers_by_type(provider_type)
            
            if args.format == 'json':
                print(json.dumps(providers, indent=2))
            else:
                # Table format
                print(f"Providers of type '{args.provider_type}':")
                print("-" * (len(args.provider_type) + 20))
                for provider_id in providers:
                    print(f"- {provider_id}")
        
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    elif args.discover_command == 'by-capability':
        try:
            # Convert string to ProviderCapability enum
            capability = args.capability
            for cap in ProviderCapability:
                if cap.value == capability:
                    capability = cap
                    break
            
            providers = ProviderDiscovery.find_providers_by_capability(capability)
            
            if args.format == 'json':
                print(json.dumps(providers, indent=2))
            else:
                # Table format
                print(f"Providers with capability '{args.capability}':")
                print("-" * (len(args.capability) + 25))
                for provider_id in providers:
                    print(f"- {provider_id}")
        
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    elif args.discover_command == 'example':
        try:
            example = ProviderDiscovery.create_provider_example(args.provider_id)
            print(example)
        
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    elif args.discover_command == 'capabilities':
        capabilities = ProviderDiscovery.get_available_capabilities()
        
        if args.format == 'json':
            print(json.dumps(capabilities, indent=2))
        else:
            # Table format
            print("Available Provider Capabilities:")
            print("-------------------------------")
            for capability in capabilities:
                print(f"- {capability}")
    
    else:
        print("Unknown discovery command", file=sys.stderr)
        return 1
    
    return 0