import { useQuery } from '@tanstack/react-query';
import { mockTarsApi } from '../services/tarsApi';

export const useAgents = () => {
  return useQuery({
    queryKey: ['agents'],
    queryFn: mockTarsApi.getAgents,
    refetchInterval: 3000, // Refetch every 3 seconds for real-time agent status
  });
};
